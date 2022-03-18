import { Box, Button, TextField, Typography, useTheme } from "@mui/material"
import React, { FC, createRef, useState, useEffect } from "react"
import LoadingButton from "@mui/lab/LoadingButton"
import SaveIcon from "@mui/icons-material/Save"

import { actionCreator } from "../action"

export const Note: FC<{
  studyId: number
  latestNote: Note
}> = ({ studyId, latestNote }) => {
  const theme = useTheme()
  const [saving, setSaving] = useState(false)
  const [disable, setDisable] = useState(true)
  const [curNote, setCurNote] = useState({ version: 0, body: "" })
  const textAreaRef = createRef<HTMLTextAreaElement>()
  const action = actionCreator()
  const notLatest = latestNote.version > curNote.version

  useEffect(() => {
    setCurNote(latestNote)
  }, [])
  const handleSave = () => {
    const nextVersion = curNote.version + 1
    const newNote = {
      version: nextVersion,
      body: textAreaRef.current ? textAreaRef.current.value : "",
    }
    setSaving(true)
    action
      .saveNote(studyId, newNote)
      .then(() => {
        setCurNote(newNote)
      })
      .finally(() => {
        setSaving(false)
      })
  }
  const handleRefresh = () => {
    if (!textAreaRef.current) {
      console.log("Unexpectedly, textarea is not found.")
      return
    }
    textAreaRef.current.value = latestNote.body
    setCurNote(latestNote)
  }

  return (
    <>
      <TextField
        disabled={saving}
        minRows={10}
        multiline={true}
        placeholder="Take a note (The note is saved to study's system_attrs)"
        sx={{ width: "100%", margin: `${theme.spacing(1)} 0` }}
        inputRef={textAreaRef}
        defaultValue={curNote.body}
        onChange={() => {
          const cur = textAreaRef.current ? textAreaRef.current.value : ""
          setDisable(cur === curNote.body)
        }}
      />
      <Box sx={{ display: "flex", flexDirection: "row", alignItems: "center" }}>
        {notLatest && !saving && (
          <>
            <Typography
              sx={{
                color: theme.palette.error.main,
                fontSize: "0.8rem",
                display: "inline",
              }}
            >
              The text you are editing has updated. Do you want to discard your
              changes and refresh the textarea?
            </Typography>
            <Button
              variant="text"
              onClick={handleRefresh}
              color="error"
              size="small"
              sx={{ textDecoration: "underline" }}
            >
              Yes
            </Button>
          </>
        )}
        <Box sx={{ flexGrow: 1 }} />
        <LoadingButton
          onClick={handleSave}
          loading={saving}
          loadingPosition="start"
          startIcon={<SaveIcon />}
          variant="contained"
          disabled={disable}
        >
          Save
        </LoadingButton>
      </Box>
    </>
  )
}
