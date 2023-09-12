import React, { FC, useState } from "react"
import {
  Typography,
  Box,
  useTheme,
  Card,
  CardContent,
  CardActions,
  CardActionArea,
} from "@mui/material"
import ClearIcon from "@mui/icons-material/Clear"
import IconButton from "@mui/material/IconButton"
import OpenInFullIcon from "@mui/icons-material/OpenInFull"
import ReplayIcon from "@mui/icons-material/Replay"
import Modal from "@mui/material/Modal"
import { red } from "@mui/material/colors"
import UndoIcon from "@mui/icons-material/Undo"

import { actionCreator } from "../action"
import { TrialListDetail } from "./TrialList"
import { MarkdownRenderer } from "./Note"

const PreferentialTrial: FC<{
  trial?: Trial
  candidates: number[]
  hideTrial: () => void
}> = ({ trial, candidates, hideTrial }) => {
  const theme = useTheme()
  const action = actionCreator()
  const trialWidth = 500
  const trialHeight = 300
  const [detailShown, setDetailShown] = useState(false)

  if (trial == undefined) {
    return (
      <Box
        sx={{
          width: trialWidth,
          minHeight: trialHeight,
          margin: theme.spacing(2),
        }}
      />
    )
  }

  const isBestTrial = trial.state === "Complete"

  return (
    <Card
      sx={{
        width: trialWidth,
        minHeight: trialHeight,
        margin: theme.spacing(2),
        padding: 0,
      }}
    >
      <CardActions>
        <Typography variant="h5">Trial {trial.number}</Typography>
        <IconButton
          sx={{
            marginLeft: "auto",
          }}
          onClick={() => {
            hideTrial()
            action.skipPreferentialTrial(trial.study_id, trial.trial_id)
          }}
          aria-label="skip trial"
        >
          <ReplayIcon />
        </IconButton>
        <IconButton
          sx={{
            marginLeft: "auto",
          }}
          onClick={() => setDetailShown(true)}
          aria-label="show detail"
        >
          <OpenInFullIcon />
        </IconButton>
      </CardActions>
      <CardActionArea>
        <CardContent
          aria-label="trial-button"
          onClick={() => {
            hideTrial()
            action.updatePreference(trial.study_id, candidates, trial.number)
          }}
          sx={{
            padding: 0,
            position: "relative",
            overflow: "hidden",
            "::before": {
              content: '""',
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              backgroundColor:
                theme.palette.mode === "dark" ? "white" : "black",
              opacity: 0,
              zIndex: 1,
              transition: "opacity 0.3s ease-out",
            },
            ":hover::before": {
              opacity: 0.2,
            },
          }}
        >
          <Box
            sx={{
              padding: theme.spacing(2),
            }}
          >
            <MarkdownRenderer body={trial.note.body} />
          </Box>

          <ClearIcon
            sx={{
              position: "absolute",
              width: "100%",
              height: "100%",
              top: 0,
              left: 0,
              color: red[600],
              opacity: 0,
              transition: "opacity 0.3s ease-out",
              zIndex: 1,
              ":hover": {
                opacity: 0.3,
                filter:
                  theme.palette.mode === "dark"
                    ? "brightness(1.1)"
                    : "brightness(1.7)",
              },
            }}
          />
        </CardContent>
      </CardActionArea>
      <Modal open={detailShown} onClose={() => setDetailShown(false)}>
        <Box
          sx={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            width: "80%",
            maxHeight: "90%",
            margin: "auto",
            overflow: "hidden",
            backgroundColor: theme.palette.mode === "dark" ? "black" : "white",
            borderRadius: theme.spacing(3),
          }}
        >
          <Box
            sx={{
              width: "100%",
              height: "100%",
              overflow: "auto",
            }}
          >
            <TrialListDetail
              trial={trial}
              isBestTrial={() => isBestTrial}
              directions={[]}
              objectiveNames={[]}
            />
          </Box>
        </Box>
      </Modal>
    </Card>
  )
}

type DisplayTrials = {
  display: number[]
  clicked: number[]
}

export const PreferentialTrials: FC<{ studyDetail: StudyDetail | null }> = ({
  studyDetail,
}) => {
  const [undoHistoryId, setUndoHistoryId] = useState<string | null>(null)

  if (studyDetail === null || !studyDetail.is_preferential) {
    return null
  }
  const theme = useTheme()
  const action = actionCreator()

  const runningTrials = studyDetail.trials.filter((t) => t.state === "Running")
  const activeTrials = runningTrials.concat(studyDetail.best_trials)

  const [displayTrials, setDisplayTrials] = useState<DisplayTrials>({
    display: [],
    clicked: [],
  })
  const new_trails = activeTrials.filter(
    (t) =>
      !displayTrials.display.includes(t.number) &&
      !displayTrials.clicked.includes(t.number)
  )
  if (new_trails.length > 0) {
    setDisplayTrials((prev) => {
      const display = [...prev.display]
      const clicked = [...prev.clicked]
      new_trails.map((t) => {
        const index = display.findIndex((n) => n === -1)
        if (index === -1) {
          display.push(t.number)
          clicked.push(-1)
        } else {
          display[index] = t.number
        }
      })
      return {
        display: display,
        clicked: clicked,
      }
    })
  }

  const hideTrial = (num: number) => {
    setDisplayTrials((prev) => {
      const index = prev.display.findIndex((n) => n === num)
      if (index === -1) {
        return prev
      }
      const display = [...prev.display]
      const clicked = [...prev.clicked]
      display[index] = -1
      clicked[index] = num
      return {
        display: display,
        clicked: clicked,
      }
    })
  }
  const latestHistoryId =
    studyDetail?.preference_history?.filter((h) => !h.isRemoved).pop()?.id ??
    null
  if (undoHistoryId !== null && undoHistoryId !== latestHistoryId) {
    setUndoHistoryId(latestHistoryId)
  }

  return (
    <Box padding={theme.spacing(2)}>
      <Box display="flex">
        <Typography
          variant="h4"
          sx={{
            marginBottom: theme.spacing(2),
            fontWeight: theme.typography.fontWeightBold,
          }}
        >
          Which trial is the worst?
        </Typography>
        <IconButton
          disabled={latestHistoryId === null || undoHistoryId !== null}
          onClick={() => {
            if (latestHistoryId === null) {
              return
            }
            setUndoHistoryId(latestHistoryId)
            action.removePreferentialHistory(studyDetail.id, latestHistoryId)
          }}
          sx={{
            margin: "auto 0 auto auto",
          }}
        >
          <UndoIcon />
        </IconButton>
      </Box>
      <Box sx={{ display: "flex", flexDirection: "row", flexWrap: "wrap" }}>
        {displayTrials.display.map((t, index) => (
          <PreferentialTrial
            key={index}
            trial={activeTrials.find((trial) => trial.number === t)}
            candidates={displayTrials.display.filter((n) => n !== -1)}
            hideTrial={() => {
              hideTrial(t)
            }}
          />
        ))}
      </Box>
    </Box>
  )
}
