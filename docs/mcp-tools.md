# Kaggle 官方 MCP 工具清单（实时抓取）

- 端点: `https://www.kaggle.com/mcp`
- 工具数: **70**
- 抓取方式: `tools/list` + Bearer KGAT token

## 按类别

### auth_quota (3)

- **`authorize`** — This tool can be used to authorize with Kaggle but it is only compatible with certain clients. See kaggle.com/docs/mcp for more information
- **`get_accelerator_quota`** — Get the current user's weekly GPU and TPU quota usage.
- **`get_user_profile`** — Get a user's public profile.

### competition (20)

- **`create_code_competition_submission`** — Submit a kernel to a competition.
- **`download_competition_data_file`** — Download a single data file for a competition.
- **`download_competition_data_files`** — Download all data files for a competition.
- **`download_competition_leaderboard`** — Download the leaderboard for a competition.
- **`get_competition`** — Get competition metadata.
- **`get_competition_data_files_summary`** — Get a summary of the data files for a competition, including number and type of files as well as column metadata.
- **`get_competition_leaderboard`** — Get the leaderboard for a competition.
- **`get_competition_submission`** — Get metadata about a competition submission.
- **`get_episode_agent_logs`** — Download agent logs for a simulation episode.
- **`get_episode_replay`** — Download the replay for a simulation episode.
- **`list_competition_data_files`** — List the data files for a competition.
- **`list_competition_data_tree_files`** — List the data files by directory for a competition.
- **`list_competition_pages`** — List the content pages for a competition (overview, rules, evaluation, data description, etc.).
- **`list_competition_topics`** — List discussion topics for a competition.
- **`list_submission_episodes`** — List episodes for a submission in a simulation competition.
- **`list_team_public_submissions`** — List the public-safe submissions for a team. For simulation competitions this is every active (leaderboard-eligible) submission; for regular competitions it is the single submission currently on the public leaderboard.
- **`search_competition_submissions`** — Search for competition submissions.
- **`search_competitions`** — Search for competitions.
- **`start_competition_submission_upload`** — Get a link to upload a competition submission file.
- **`submit_to_competition`** — Submit a new competition submission.

### dataset (10)

- **`download_dataset`** — Download a dataset.
- **`get_dataset_files_summary`** — Get a summary of the data files in a dataset, including number and type of files as well as column metadata.
- **`get_dataset_info`** — Get information about a dataset.
- **`get_dataset_metadata`** — Get metadata about a dataset.
- **`get_dataset_status`** — Get the processing status of a dataset, indicating whether the dataset is still processing, completed, or failed.
- **`list_dataset_files`** — List the files in a dataset.
- **`list_dataset_tree_files`** — List the files in a dataset by directory.
- **`search_datasets`** — Search for datasets.
- **`update_dataset_metadata`** — Update the metadata for a dataset.
- **`upload_dataset_file`** — Upload a file to a dataset.

### notebook (10)

- **`cancel_notebook_session`** — Cancel a notebook session.
- **`create_notebook_session`** — Create a new notebook session.
- **`download_notebook_output`** — Download the output of a notebook.
- **`download_notebook_output_zip`** — Download the zipped output of a notebook.
- **`get_notebook_info`** — Get information about a notebook.
- **`get_notebook_session_status`** — Get the status of a notebook session, indicating whether it is running, completed, or failed..
- **`list_notebook_files`** — List the files in a notebook.
- **`list_notebook_session_output`** — List the output files of a notebook session.
- **`save_notebook`** — Save a notebook and run it from top to bottom.
- **`search_notebooks`** — Search for notebooks.

### model (10)

- **`create_model`** — Create a new model.
- **`download_model_variation_version`** — Get a url where the files for a model variation version can be downloaded.
- **`get_model`** — Get a metadata about a model.
- **`get_model_variation`** — Get metadata about a model variation.
- **`list_model_variation_version_files`** — List the files of a model variation version.
- **`list_model_variation_versions`** — List model variation versions.
- **`list_model_variations`** — List model variations.
- **`list_models`** — List models.
- **`update_model`** — Update a model.
- **`update_model_variation`** — Update a model variation.

### forum_writeup (14)

- **`download_hackathon_write_ups`** — Exports a CSV with the data from all submitted writeups after the hackathon has closed.
- **`get_forum`** — Get details about a discussion forum, including its name, description, and topic count.
- **`get_forum_topic`** — Get a discussion topic by ID, optionally including its comments.
- **`get_hackathon_overview`** — Get all the overview page content for a hackathon competition.
- **`get_hackathon_write_up`** — Get a single hackathon write-up submission by ID for a competition.
- **`get_resolved_writeup_links`** — Get resolved information for all links in a writeup, including download URLs, file summaries, and metadata for datasets, notebooks, YouTube videos, and external links.
- **`get_writeup`** — Get a writeup by ID.
- **`get_writeup_by_slug`** — Get a writeup by its slug.
- **`get_writeup_by_topic`** — Get a writeup by its linked forum topic ID. Use this when you found a writeup via forum discussions.
- **`list_forum_topics`** — List and search discussion topics. Omit forum_id to search across all forums.
- **`list_forums`** — List all top-level discussion forums on Kaggle.
- **`list_hackathon_tracks`** — List the tracks for a hackathon competition.
- **`list_hackathon_write_ups`** — List hackathon write-up submissions for a competition. Supports filtering by track and winner status.
- **`list_topic_messages`** — List discussion messages within a competition topic.

### benchmark (2)

- **`create_benchmark_task_from_prompt`** — Create a new benchmark task from a prompt.
- **`get_benchmark_leaderboard`** — Get the leaderboard for a benchmark.

### search (1)

- **`search_content`** — Search across all Kaggle content types (competitions, datasets, notebooks, models, discussions, users, benchmarks).
