# mini-support-ai

## Step 3: K-means and TF-IDF results

### With num_clusters=5
--- Sample Questions per Cluster ---

Cluster 0:
  - why ternary operator in swift is so picky?
  - why does the reverse() function in the swift standard library return reverserandomaccesscollection?
  - can i throw from class init() in swift with constant string loaded from file?

Cluster 1:
  - text overlay image with darkened opacity react native
  - how to delete compiled js files from previous typescript(.ts) files?
  - es6 - import all named module without alias

Cluster 2:
  - why are java optionals immutable?
  - hide/show fab with scale animation
  - changing theme in windows 10 uwp app programmatically

Cluster 3:
  - how to play gif in android from url?
  - in android studio 2.0, cannot find local variable of method in debug mode
  - cannot resolve symbol 'requestqueue'

Cluster 4:
  - disable logging for one container in docker-compose
  - persist elastic search data in docker container
  - redirecting command output in docker

--- Top Terms per Cluster ---

Cluster 0 top terms:
  string
  class
  type
  public
  return
  function
  int
  new
  value
  var

Cluster 1 top terms:
  react
  component
  js
  div
  import
  app
  angular
  native
  webpack
  props

Cluster 2 top terms:
  file
  using
  error
  use
  like
  code
  app
  way
  want
  does

Cluster 3 top terms:
  android
  com
  studio
  java
  app
  gradle
  support
  build
  project
  id

Cluster 4 top terms:
  docker
  compose
  container
  image
  run
  dockerfile
  build
  containers
  yml
  running


  ### With num_clusters=6
  --- Sample Questions per Cluster ---

Cluster 0:
  - why ternary operator in swift is so picky?
  - why does the reverse() function in the swift standard library return reverserandomaccesscollection?
  - can i throw from class init() in swift with constant string loaded from file?

Cluster 1:
  - text overlay image with darkened opacity react native
  - how to delete compiled js files from previous typescript(.ts) files?
  - es6 - import all named module without alias

Cluster 2:
  - hide/show fab with scale animation
  - changing theme in windows 10 uwp app programmatically
  - mongodb failing to start - ***aborting after fassert() failure

Cluster 3:
  - why are java optionals immutable?
  - java mongodb driver how do you catch exceptions?
  - visualvm fails with "no jdkhome found" on ubuntu 15.10 with oracle jdk

Cluster 4:
  - disable logging for one container in docker-compose
  - persist elastic search data in docker container
  - redirecting command output in docker

Cluster 5:
  - how to play gif in android from url?
  - in android studio 2.0, cannot find local variable of method in debug mode
  - cannot resolve symbol 'requestqueue'

--- Top Terms per Cluster ---

Cluster 0 top terms:
  string
  class
  type
  public
  return
  function
  int
  new
  value
  var

Cluster 1 top terms:
  react
  component
  js
  div
  import
  app
  angular
  native
  props
  webpack

Cluster 2 top terms:
  file
  using
  use
  error
  like
  app
  code
  way
  want
  does

Cluster 3 top terms:
  java
  df
  dataframe
  pandas
  column
  date
  org
  np
  data
  00

Cluster 4 top terms:
  docker
  compose
  container
  image
  run
  dockerfile
  build
  containers
  yml
  running

Cluster 5 top terms:
  android
  com
  studio
  app
  java
  gradle
  support
  build
  project
  id


  ### With num_clusters=7
  Cluster 0:
  - why ternary operator in swift is so picky?
  - why does the reverse() function in the swift standard library return reverserandomaccesscollection?
  - can i throw from class init() in swift with constant string loaded from file?

Cluster 1:
  - text overlay image with darkened opacity react native
  - how to delete compiled js files from previous typescript(.ts) files?
  - es6 - import all named module without alias

Cluster 2:
  - hide/show fab with scale animation
  - changing theme in windows 10 uwp app programmatically
  - mongodb failing to start - ***aborting after fassert() failure

Cluster 3:
  - why are java optionals immutable?
  - java mongodb driver how do you catch exceptions?
  - visualvm fails with "no jdkhome found" on ubuntu 15.10 with oracle jdk

Cluster 4:
  - django: attributeerror: 'nonetype' object has no attribute 'split'
  - nodejs express encodes the url - how to decode
  - are there any alternatives to t4 templates and envdte for cross platform asp.net 5 development?

Cluster 5:
  - how to play gif in android from url?
  - in android studio 2.0, cannot find local variable of method in debug mode
  - cannot resolve symbol 'requestqueue'

Cluster 6:
  - disable logging for one container in docker-compose
  - persist elastic search data in docker container
  - redirecting command output in docker

--- Top Terms per Cluster ---

Cluster 0 top terms:
  string
  class
  int
  type
  public
  return
  std
  foo
  new
  void

Cluster 1 top terms:
  react
  component
  js
  div
  import
  app
  native
  angular
  props
  webpack

Cluster 2 top terms:
  file
  using
  use
  like
  error
  code
  way
  want
  python
  does

Cluster 3 top terms:
  java
  df
  dataframe
  pandas
  org
  column
  spark
  pd
  columns
  apache

Cluster 4 top terms:
  request
  net
  api
  user
  http
  app
  response
  core
  server
  error

Cluster 5 top terms:
  android
  com
  studio
  app
  gradle
  java
  support
  build
  project
  id

Cluster 6 top terms:
  docker
  compose
  container
  image
  run
  dockerfile
  build
  containers
  yml
  running


  ## STEP - 4
🧠 Why Step 4: Embedding + Semantic Search?
🎯 Problem with Current Setup (KMeans + TF-IDF):
TF-IDF is keyword-based: it can’t understand meaning or similarity if the words don’t match exactly.

Example:

User searches: “git push fails after rebase”

Your TF-IDF model may not match: “unable to push branch due to upstream mismatch”

❌ TF-IDF doesn’t "know" those are related.
✅ That’s where semantic search comes in.

💡 What is Semantic Search?
Semantic search finds results based on meaning, not just word match.

Example:
Query: "can't pull docker image from private registry"

Match: "Artifactory Docker authentication fails with 403"

Even though the words are different, they’re semantically similar.

🔍 What Are Embeddings?
An embedding is a vector (list of numbers) that represents the meaning of a sentence.

For example:

"docker pull fails" → [0.12, -0.03, ..., 0.98] (384-dimensional vector)

"can't download image from registry" → a very similar vector

So now you can compare meaning using math (cosine distance, L2, etc.)

🧠 What is BERT (and Sentence-BERT)?
BERT = Bidirectional Encoder Representations from Transformers

It’s a powerful language model that understands sentence meaning

Sentence-BERT (S-BERT) = A tuned version that outputs embeddings for entire sentences, not just words

We’ll use this model:

python
Copy
Edit
sentence-transformers/all-MiniLM-L6-v2
✅ Lightweight
✅ Fast
✅ Good accuracy

📦 What is FAISS?
FAISS = Facebook AI Similarity Search
It’s a vector database that:

Stores your embeddings (all 15k questions)

Quickly finds most similar entries to a user’s query

Instead of searching by keyword, you search by meaning.

🔁 Summary: What Step 4 Will Do
Step	Action
✅ 1	Embed each question (title + body) using Sentence-BERT
✅ 2	Store all those vectors in FAISS index
✅ 3	Accept user queries, embed them too
✅ 4	Use FAISS to return the most similar questions from the dataset
✅ 5	Show result with original title + answer + cluster label (from Step 3)

