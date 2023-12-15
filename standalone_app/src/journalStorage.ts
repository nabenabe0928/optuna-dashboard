import { SetterOrUpdater } from "recoil"

// JournalStorage
enum JournalOperation {
  CREATE_STUDY = 0,
  DELETE_STUDY = 1,
  SET_STUDY_USER_ATTR = 2,
  SET_STUDY_SYSTEM_ATTR = 3,
  CREATE_TRIAL = 4,
  SET_TRIAL_PARAM = 5,
  SET_TRIAL_STATE_VALUES = 6,
  SET_TRIAL_INTERMEDIATE_VALUE = 7,
  SET_TRIAL_USER_ATTR = 8,
  SET_TRIAL_SYSTEM_ATTR = 9,
}

interface JournalOpBase {
  op_code: JournalOperation
  workor_id: string
}

interface JournalOpCreateStudy extends JournalOpBase {
  study_name: string
  directions: number[] // TODO(gen740): introduce Study Direction enum
}

interface JournalOpDeleteStudy extends JournalOpBase {
  study_id: number
}

interface JournalOpCreateTrial extends JournalOpBase {
  study_id: number
  datetime_start?: string
  datetime_complete?: string
  distributions?: { [key: string]: string }
  params?: { [key: string]: any }
  user_attrs?: { [key: string]: any }
  system_attrs?: { [key: string]: any }
  state?: number
  intermediate_values?: { [key: string]: number }
  value?: number
  values?: number[]
}

interface JournalOpSetTrialParam extends JournalOpBase {
  trial_id: number
  param_name: string
  param_value_internal: number
  distribution: string
}

interface JournalOpSetTrialStateValue extends JournalOpBase {
  trial_id: number
  state: number
  values?: number[]
  datetime_start?: string
  datetime_complete?: string
}

interface JournalOpSetTrialIntermediateValue extends JournalOpBase {
  trial_id: number
  step: number
  intermediate_value: number
}

interface JournalOpSetTrialUserAttr extends JournalOpBase {
  trial_id: number
  user_attr: { [key: string]: any }
}

const trialStateNumToTrialState = (state: number): TrialState => {
  switch (state) {
    case 0:
      return "Running"
    case 1:
      return "Complete"
    case 2:
      return "Pruned"
    case 3:
      return "Fail"
    case 4:
      return "Waiting"
    default:
      return "Running"
  }
}

const parseDistribution = (distribution: string): Distribution => {
  const distributionJson = JSON.parse(distribution)
  if (distributionJson["name"] === "IntDistribution") {
    return {
      ...distributionJson["attributes"],
      type: "IntDistribution",
    }
  } else if (distributionJson["name"] === "FloatDistribution") {
    return {
      ...distributionJson["attributes"],
      type: "FloatDistribution",
    }
  } else {
    return {
      // TODO(gen740): support other types
      type: "CategoricalDistribution",
      choices: distributionJson["attributes"]["choices"].map((choice: any) => {
        return {
          pytype: "str",
          value: choice.toString(),
        }
      }),
    }
  }
}

class JournalStorage {
  private studies: Study[] = []
  private nextStudyId = 0
  private studyIdToTrialIDs: Map<number, number[]> = new Map()
  private trialIdToStudyId: Map<number, number> = new Map()
  private trialID = 0

  public getStudies(): Study[] {
    for (let study of this.studies) {
      let unionUserAttrs: Set<string> = new Set()
      let unionSearchSpace: Set<string> = new Set()
      let intersectionSearchSpace: string[] = []

      study.trials.forEach((trial, index) => {
        for (let userAttr of trial.user_attrs) {
          console.log(userAttr.key)
          unionUserAttrs.add(userAttr.key)
        }
        for (let param of trial.params) {
          unionSearchSpace.add(param.name)
        }
        if (index === 0) {
          intersectionSearchSpace = Array.from(unionSearchSpace)
        } else {
          intersectionSearchSpace = intersectionSearchSpace.filter((name) => {
            return trial.params.some((param) => param.name === name)
          })
        }
      })
      study.union_user_attrs = Array.from(unionUserAttrs).map((key) => {
        return {
          key: key,
          sortable: false,
        }
      })
      study.union_search_space = Array.from(unionSearchSpace).map((name) => {
        return {
          name: name,
        }
      })
      study.intersection_search_space = intersectionSearchSpace.map((name) => {
        return {
          name: name,
        }
      })
    }

    return this.studies
  }

  public applyCreateStudy(log: JournalOpCreateStudy): void {
    this.studies.push({
      study_id: this.nextStudyId,
      study_name: log.study_name,
      directions: [log.directions[0] === 1 ? "minimize" : "maximize"],
      union_search_space: [],
      intersection_search_space: [],
      union_user_attrs: [],
      trials: [],
    })
    this.nextStudyId++
  }

  public applyDeleteStudy(log: JournalOpDeleteStudy): void {
    this.studies = this.studies.filter((item) => item.study_id != log.study_id)
  }

  public applyCreateTrial(log: JournalOpCreateTrial): void {
    let thisStudy = this.studies.find((item) => item.study_id == log.study_id)
    if (thisStudy === undefined) {
      return
    }

    let params: TrialParam[] =
      log.params === undefined
        ? []
        : Object.entries(log.params).map(([name, value]) => {
            const distribution = parseDistribution(log.distributions![name])
            return {
              name: name,
              param_internal_value: value,
              param_external_type: distribution.type,
              param_external_value: (() => {
                if (distribution.type === "FloatDistribution") {
                  return value.toString()
                } else if (distribution.type === "IntDistribution") {
                  return value.toString()
                } else {
                  return distribution.choices[value].value
                }
              })(),
              distribution: distribution,
            }
          })

    const userAtter = log.user_attrs
      ? Object.entries(log.user_attrs).map(([key, value]) => {
          return {
            key: key,
            value: value,
          }
        })
      : []

    thisStudy.trials.push({
      trial_id: this.trialID,
      number: this.studyIdToTrialIDs.get(log.study_id)?.length ?? 0,
      study_id: log.study_id,
      state: trialStateNumToTrialState(log.state ?? 0),
      values: (() => {
        if (log.value !== undefined) {
          return [log.value]
        } else if (log.values !== undefined) {
          return log.values
        } else {
          return undefined
        }
      })(),
      params: params,
      intermediate_values: [],
      user_attrs: userAtter,
      datetime_start: log.datetime_start
        ? new Date(log.datetime_start)
        : undefined,
      datetime_complete: log.datetime_complete
        ? new Date(log.datetime_complete)
        : undefined,
    })
    this.studyIdToTrialIDs.set(
      log.study_id,
      this.studyIdToTrialIDs.get(log.study_id)?.concat([this.trialID]) ?? [
        this.trialID,
      ]
    )
    this.trialIdToStudyId.set(this.trialID, log.study_id)
    this.trialID++
  }

  private getStudyAndTrial(trial_id: number): [Study?, Trial?] {
    let study = this.studies.find(
      (item) => item.study_id == this.trialIdToStudyId.get(trial_id)
    )
    if (study === undefined) {
      return [undefined, undefined]
    }

    const trial = study.trials.find((item) => item.trial_id == trial_id)
    if (trial === undefined) {
      return [study, undefined]
    }
    return [study, trial]
  }

  public applySetTrialParam(log: JournalOpSetTrialParam) {
    let [thisStudy, thisTrial] = this.getStudyAndTrial(log.trial_id)
    if (thisStudy === undefined || thisTrial === undefined) {
      return
    }
    thisTrial.params.push({
      name: log.param_name,
      param_internal_value: log.param_value_internal,
      param_external_type: "FloatDistribution",
      param_external_value: log.param_value_internal.toString(),
      distribution: parseDistribution(log.distribution),
    })
  }

  public applySetTrialStateValues(log: JournalOpSetTrialStateValue): void {
    let [thisStudy, thisTrial] = this.getStudyAndTrial(log.trial_id)
    if (thisStudy === undefined || thisTrial === undefined) {
      return
    }
    thisTrial.state = trialStateNumToTrialState(log.state)
    thisTrial.values = log.values
    thisTrial.datetime_start = log.datetime_start
      ? new Date(log.datetime_start)
      : undefined
    thisTrial.datetime_complete = log.datetime_complete
      ? new Date(log.datetime_complete)
      : undefined
  }

  public applySetTrialIntermediateValue(
    log: JournalOpSetTrialIntermediateValue
  ) {
    let [thisStudy, thisTrial] = this.getStudyAndTrial(log.trial_id)
    if (thisStudy === undefined || thisTrial === undefined) {
      return
    }
    thisTrial.intermediate_values.push({
      step: log.step,
      value: log.intermediate_value,
    })
  }

  public applySetTrialUserAttr(log: JournalOpSetTrialUserAttr) {
    let [thisStudy, thisTrial] = this.getStudyAndTrial(log.trial_id)
    if (thisStudy === undefined || thisTrial === undefined) {
      return
    }
    for (let [key, value] of Object.entries(log.user_attr)) {
      thisTrial.user_attrs.push({
        key: key,
        value: value.toString(),
      })
    }
  }
}

export const loadJournalStorage = (
  arrayBuffer: ArrayBuffer,
  setter: SetterOrUpdater<Study[]>
): void => {
  const decoder = new TextDecoder("utf-8")
  const logs = decoder.decode(arrayBuffer).split("\n")

  let journalStorage = new JournalStorage()

  for (let log of logs) {
    if (log === "") {
      continue
    }
    let parsedLog: JournalOpBase = JSON.parse(log)
    switch (parsedLog.op_code) {
      case JournalOperation.CREATE_STUDY:
        journalStorage.applyCreateStudy(parsedLog as JournalOpCreateStudy)
        break
      case JournalOperation.DELETE_STUDY:
        journalStorage.applyDeleteStudy(parsedLog as JournalOpDeleteStudy)
        break
      case JournalOperation.SET_STUDY_USER_ATTR:
        // Unsupported set for study user_attr
        break
      case JournalOperation.SET_STUDY_SYSTEM_ATTR:
        // Unsupported set for study system_attr
        break
      case JournalOperation.CREATE_TRIAL:
        journalStorage.applyCreateTrial(parsedLog as JournalOpCreateTrial)
        break
      case JournalOperation.SET_TRIAL_PARAM:
        journalStorage.applySetTrialParam(parsedLog as JournalOpSetTrialParam)
        break
      case JournalOperation.SET_TRIAL_STATE_VALUES:
        journalStorage.applySetTrialStateValues(
          parsedLog as JournalOpSetTrialStateValue
        )
        break
      case JournalOperation.SET_TRIAL_INTERMEDIATE_VALUE:
        journalStorage.applySetTrialIntermediateValue(
          parsedLog as JournalOpSetTrialIntermediateValue
        )
        break
      case JournalOperation.SET_TRIAL_USER_ATTR:
        journalStorage.applySetTrialUserAttr(
          parsedLog as JournalOpSetTrialUserAttr
        )
        break
      case JournalOperation.SET_TRIAL_SYSTEM_ATTR:
        // Unsupported set for trial system_attr
        break
    }
  }

  console.log(journalStorage.getStudies())
  setter((prev) => [...prev, ...journalStorage.getStudies()])
}
