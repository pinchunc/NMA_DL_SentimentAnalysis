# Hippocorpus V3
This directory contains the files part of the V3 version of the Hippocorpus dataset.
As with previous versions,`hc-stories.csv` contains the imagined/recalled/retold stories, summaries, and worker characteristics.
*New in V3*:
A subset of 240 stories were annotated for whether each sentence has an event or not (in `hc-eventAnnots.csv` and  `hc-eventAnnotsAggOverWorkers.csv`).

## Hippocorpus stories `hc-stories.csv`
This file contains the stories themselves.
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


## Hippocorpus event annotations (`hc-eventAnnots.csv`, `hc-eventAnnotsAggOverWorkers.csv`)
The files contain the following columns:
- `HITId`: Unique ID for the set of three stories.
- `summary`: Summary of the story.
- `AssignmentId`: Unique ID that identifies the set of stories and the worker who annotated them.
- `WorkerId`: Unique ID that identifies a worker (anonymized). Note: only in `hc-eventAnnots.csv`, this was grouped over to create `hc-eventAnnotsAggOverWorkers.csv`.
- `rawEventAnnot`: Raw event annotation.
- `storyIx`: The order in which the story was presented to the worker (1, 2, 3).
- `sentIx`: The sentence index in the story.
- `memType`: Whether the story was recalled, retold, or imagined.
- `sent`: Actual sentence text.
- `majorBin`: binarized indicator variable for whether the sentence was annotated to contain a *major* event.
- `minorBin`: binarized indicator variable for whether the sentence was annotated to contain a *minor* event.
- `expectedBin`: binarized indicator variable for whether the sentence was annotated to contain an *expected* event.
- `surprisingBin`: binarized indicator variable for whether the sentence was annotated to contain a *surprising* event.
- `eventOrNot`: binarized indicator variable for whether the sentence was annotated to contain an event or not.
- `HippocorpusAssignmentId`: original story ID from the Hippocorpus, can be used to tie back to the variables in that file.


## Citation:
**Citation for the Hippocorpus**
Maarten Sap, Eric Horvitz, Yejin Choi, Noah A. Smith, and James Pennebaker (2020) _Recollection versus Imagination: Exploring Human Memory and Cognition via Neural Language Models._ ACL.
