This CSV file contains all the stories in Hippcorpus v2 (6854 stories)

Breakdown by type of story:
```
recalled    2779
imagined    2756
retold      1319
Name: memType, dtype: int64
```

These are the columns in the file:
- `AssignmentId`: Unique ID of this story
- `WorkTimeInSeconds`: Time in seconds that it took the worker to do the entire HIT (reading instructions, storywriting, questions)
- `WorkerId`: Unique ID of the worker (random string, not MTurk worker ID)
- `annotatorAge`: Lower limit of the age bucket of the worker. Buckets are: 18-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55+
- `annotatorGender`: Gender of the worker
- `annotatorRace`: Race/ethnicity of the worker
- `distracted`: How distracted were you while writing your story? (5-point Likert)
- `draining`: How taxing/draining was writing for you emotionally? (5-point Likert)
- `frequency`: How often do you think about or talk about this event? (5-point Likert)
- `importance`: How impactful, important, or personal is this story/this event to you? (5-point Likert)
- `logTimeSinceEvent`: Log of time (days) since the recalled event happened
- `mainEvent`: Short phrase describing the main event described
- `memType`: Type of story (recalled, imagined, retold)
- `mostSurprising`: Short phrase describing what the most surpring aspect of the story was
- `openness`: Continuous variable representing the openness to experience of the worker
- `recAgnPairId`: ID of the recalled story that corresponds to this retold story (null for imagined stories). Group on this variable to get the recalled-retold pairs.
- `recImgPairId`: ID of the recalled story that corresponds to this imagined story (null for retold stories). Group on this variable to get the recalled-imagined pairs.
- `similarity`: How similar to your life does this event/story feel to you? (5-point Likert)
- `similarityReason`: Free text annotation of similarity
- `story`: Story about the imagined or recalled event (15-25 sentences)
- `stressful`: How stressful was this writing task? (5-point Likert)
- `summary`: Summary of the events in the story (1-3 sentences)
- `timeSinceEvent`: Time (num. days) since the recalled event happened

**Citation**
Maarten Sap, Eric Horvitz, Yejin Choi, Noah A. Smith, and James Pennebaker (2020) _Recollection versus Imagination: Exploring Human Memory and Cognition via Neural Language Models._ ACL.
