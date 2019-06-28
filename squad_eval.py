# -*- coding: utf-8 -*-
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""SQuAD-based evaluation for deep retrieval systems.

## Setup Dependencies

This program requires python3 and the following packages to run (install with
pip install):

 - tensorflow==2.0.0-alpha0
 - tensorflow-hub
 - scipy-stack
 - sentencepiece
 - tf-sentencepiece
 - recordclass

Ideally, you should use virtualenv to create an isolated python environment:

  https://virtualenv.pypa.io/en/latest/

## Run

Download the SQuAD 1.1 data files directly from:

    https://rajpurkar.github.io/SQuAD-explorer/

The following invocation runs the evaluation on the dev file. Alternatively,
use train-v1.1.json to obtain numbers that are directly comparable to the
ones reported in the ReQA paper.

    python3 squad_eval.py --saved_model /path/to/model --squad dev-v1.1.json

It should output something like:

    index size=10250 questions=11426
    mrr=0.625 r@1=0.515 r@5=0.758 r@10=0.831

Here, mrr abbreviates mean reciprocal rank, and r@N abbreviates recall at N.
"""

import argparse
import collections
import functools
import itertools
import json
import logging
import sys
from typing import Any, Callable, Dict, Generator, List, Set, Sequence, Tuple

import numpy as np
import recordclass
import sb_sed
from scipy.stats import rankdata
import tensorflow as tf
import tensorflow_hub as hub
# import tf_sentencepiece


# The saved model signature for the query encoder.
QUERY_ENCODER_SIGNATURE = "query_encoder"

# The saved model signature for the response encoder.
RESPONSE_ENCODER_SIGNATURE = "response_encoder"


def main():
  parser = argparse.ArgumentParser(
      description="Evaluate SQuAD retrieval performance.")
  parser.add_argument("--saved_model", dest="savedmodel_dir",
                      help="Location of a TensorFlow saved model.",
                      metavar="DIR")
  parser.add_argument("--tfhub", dest="tfhub_url",
                      help="Location of a TF-Hub model.",
                      metavar="URL")
  parser.add_argument("--summary", dest="summary_dir",
                      help="Where summary files are written.",
                      metavar="DIR")
  parser.add_argument("--squad", dest="squad_filename",
                      help="Location of a JSON-formatted SQuAD input file.",
                      metavar="FILE", required=True)
  args = parser.parse_args()

  root = logging.getLogger()
  root.setLevel(logging.DEBUG)

  # Load the SQuAD JSON file into memory.
  with open(args.squad_filename) as data_file:
    data = json.load(data_file)
  logging.info("loaded %s", args.squad_filename)

  if args.savedmodel_dir:
    squad_savedmodel_eval(args.savedmodel_dir, squad_json=data)
  elif args.tfhub_url:
    squad_tfhub_eval(args.tfhub_url, squad_json=data)


class QuestionAnswerInput(recordclass.RecordClass):
  """A question, its answer, the answer context, and the question encoding."""
  query: str
  response: str
  response_context: str
  doc_id: int
  encoding: np.ndarray


class Paragraph(recordclass.RecordClass):
  """A paragraph, which serves as the unit for document-level statistics."""
  id: int
  sentences: list


class IndexedItem(recordclass.RecordClass):
  """A sentence, its context, and its encoding.

    a. response: A candidate response.
    b. context: The context in which the response is embedded.
    c. encoding: A floating-point vector that encodes the response and
                 document.
    d. doc_id: The unique document identifier.
  """
  sentence: str
  context: str
  encoding: np.ndarray
  doc_id: int


class EvalResult(recordclass.RecordClass):
  """The results of an evaluation.

    mrr: mean reciprocal rank
    recall: a function that takes a single argument, n, and returns the recall
      at n.
  """
  mrr: float
  recall: Callable[[int], float]


class SquadDataset(recordclass.RecordClass):
  """Holds encoded queries and candidates from SQuAD.

    queries: A list of QuestionAnswerInputs.
    master_index: A list of IndexedItems.
    response_index: Maps response sentences to their index or indices within the
      master index. It holds that for any sentence, s:
         master_index[response_index[s]][0] == s
  """
  queries: List[QuestionAnswerInput]
  master_index: Sequence[IndexedItem]
  response_index: Dict[str, List[int]]


def make_example(inp_text="", inp_context="", res_text="", res_context=""):
  def _bytes_feature(value: str):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  return tf.train.Example(features=tf.train.Features(
      feature={
          "inp_context": _bytes_feature(inp_context.encode("utf-8")),
          "inp_text": _bytes_feature(inp_text.encode("utf-8")),
          "res_context": _bytes_feature(res_context.encode("utf-8")),
          "res_text": _bytes_feature(res_text.encode("utf-8"))
      }))


def to_array(tf_examples: Sequence[tf.train.Example]) -> np.ndarray:
  """Create an array of serialized TF examples."""
  return np.asarray([x.SerializeToString() for x in tf_examples],
                    dtype=np.object)


def squad_savedmodel_eval(savedmodel_dir, squad_json):
  """Evaluate a QA saved model against SQuAD.

  Args:
    savedmodel_dir: The location where the saved model is located.
    squad_json: The SQuAD JSON input file that will be used for evaluation.
  """
  saved_model = tf.saved_model.load(savedmodel_dir, ["serve"])

  qencoder = saved_model.signatures[QUERY_ENCODER_SIGNATURE]
  rencoder = saved_model.signatures[RESPONSE_ENCODER_SIGNATURE]

  def wrapped_qencoder(queries: List[QuestionAnswerInput]):
    return qencoder(tf.constant(to_array(
        [make_example(inp_text=q.query) for q in queries])))["outputs"]

  def wrapped_rencoder(index: List[IndexedItem]):
    examples = [
        make_example(res_text=a.sentence, res_context=a.context) for a in index]
    return rencoder(tf.constant(to_array(examples)))["outputs"]

  _eval(_load_data(squad_json, wrapped_qencoder, wrapped_rencoder))


def squad_tfhub_eval(tfhub_url, squad_json):
  """Evaluates a QA TF-Hub module against SQuAD.

  Args:
    tfhub_url: The location where the TF-Hub module is located.
    squad_json: The SQuAD JSON input file that will be used for evaluation.
  """
  saved_model = hub.load(tfhub_url)

  qencoder = saved_model.signatures["question_encoder"]
  rencoder = saved_model.signatures[RESPONSE_ENCODER_SIGNATURE]

  def wrapped_qencoder(queries: List[QuestionAnswerInput]):
    return qencoder(input=tf.constant(
        np.asarray([q.query for q in queries])))["outputs"]

  def wrapped_rencoder(index: List[IndexedItem]):
    return rencoder(
        input=tf.constant(np.asarray([a.sentence for a in index])),
        context=tf.constant(np.asarray([a.context for a in index])))["outputs"]

  _eval(_load_data(squad_json, wrapped_qencoder, wrapped_rencoder))


def _eval(squad_ds: SquadDataset):
  """Perform evaluation of a saved model."""
  # Perform the evaluation.
  sen, doc = _chunked_eval(squad_ds)

  # Evaluation is finished. Compute the final 1@N statistic and record it.
  print("index size=%s questions=%s" % (
        len(squad_ds.master_index), len(squad_ds.queries)))
  print("[sentence] mrr=%0.3f r@1=%0.3f r@5=%0.3f r@10=%0.3f" % (
        sen.mrr, sen.recall(1), sen.recall(5), sen.recall(10)))
  print("[document] mrr=%0.3f r@1=%0.3f r@5=%0.3f r@10=%0.3f" % (
        doc.mrr, doc.recall(1), doc.recall(5), doc.recall(10)))


def _chunked_eval(squad_ds: SquadDataset) \
        -> Tuple[EvalResult, EvalResult]:
  """Evaluates 'queries', in chunks of 1000 at a time.

  Returns:
    The mean reciprocal rank metric and a function that computes recall at N.
  """
  # Construct a single large matrix of the entire master_index.
  logging.info("Constructing response matrix...")
  response_matrix = np.concatenate(
      [np.expand_dims(i.encoding, 0) for i in squad_ds.master_index], axis=0)
  logging.info("Done.")

  # Chunk the work into groups of 1000 to avoid memory blowup.
  work_chunks = _chunks(squad_ds.queries, 1000)

  sentence_ranks = {}  # type: Dict[str, int]
  doc_ranks = {}  # type: Dict[str, int]
  bound_evaluate_queries = functools.partial(
      _eval_queries, squad_ds, response_matrix, sentence_ranks, doc_ranks)
  logging.info("Evaluating...")
  list(map(bound_evaluate_queries, work_chunks))

  def mrr(ranks):
    return sum([1/v for v in ranks.values()])/len(ranks)

  # A function that can compute recall at N.
  def recall_at_n(ranks, n):
    num = len([rank for rank in ranks.values() if rank <= n])
    return num / len(ranks)

  sentence_result = EvalResult(
      mrr(sentence_ranks), functools.partial(recall_at_n, sentence_ranks))
  doc_result = EvalResult(
      mrr(doc_ranks), functools.partial(recall_at_n, doc_ranks))
  return sentence_result, doc_result


def _eval_queries(squad_ds: SquadDataset,
                  response_matrix: np.ndarray,
                  sen_rank: Dict[str, int],
                  doc_rank: Dict[str, int],
                  queries: Sequence[QuestionAnswerInput]) \
        -> None:
  """Evaluate a group of queries.

  Args:
    squad_ds: A SquadDataset instance.
    response_matrix: A single matrix constructed from the encodings in the
      master_index.
    sen_rank: a string=>int dict that stores the correct answer's sentence rank.
    doc_rank: a string=>int dict that stores the correct answer's sentence rank.
    queries: A list of QuestionAnswerInputs.
  """
  queries_matrix = np.concatenate(
      [np.expand_dims(q.encoding, 0) for q in queries], axis=0)
  results_matrix = np.matmul(queries_matrix, np.transpose(response_matrix))
  for idx, qa_input in enumerate(queries):
    r = results_matrix[idx]
    ranks = rankdata(r * -1)

    # Find the rank of the correct answer.
    correct_sen_rank = min(
        [ranks[i] for i in squad_ds.response_index[qa_input.response]])
    correct_doc_rank = _eval_doc_rank(squad_ds, qa_input, ranks)

    # Update the results dict, taking care to handle the case where the question
    # has been asked before.
    def update_result(answer_rank, query, query_rank):
      answer_rank[query] = min(answer_rank.get(query, sys.maxsize), query_rank)

    update_result(sen_rank, qa_input.query, correct_sen_rank)
    update_result(doc_rank, qa_input.query, correct_doc_rank)


def _eval_doc_rank(squad_ds: SquadDataset,
                   qa_input: QuestionAnswerInput,
                   ranks: np.ndarray):
  """Given a question, evaluate its document rank.

  Given a rank for every sentence in the SquadDataset, this function first
  computes a score for every document which is defined as the rank of the
  best-ranking sentence in that document. These scores are used, in turn, to
  define a ranking among the documents. Of all the documents that contain the
  correct answer (almost always, there is only one), this method returns the
  rank of the best-ranking document.
  """
  doc_rank = {}

  # At the end of this loop, the doc_rank dictionary will contain a mapping from
  # document ids to the best-ranking answer sentence in that document.
  for j, x in enumerate(squad_ds.master_index):
    sentence_rank = ranks[j]
    document_rank = min(sentence_rank, doc_rank.get(x.doc_id, sys.maxsize))
    doc_rank[x.doc_id] = document_rank

  # The two statements below transfor the doc_rank dictionary from holding
  # sentence ranks to holding document ranks.
  kv = list(doc_rank.items())
  doc_rank = dict(zip([k for k, _ in kv], rankdata([v for k, v in kv])))

  correct_doc_ids = [squad_ds.master_index[i].doc_id
                     for i in squad_ds.response_index[qa_input.response]]
  return min([doc_rank[doc_id] for doc_id in correct_doc_ids])


def _load_data(squad_json: Any,
               qencoder: Callable[[np.ndarray], np.ndarray],
               rencoder: Callable[[np.ndarray], np.ndarray]) -> SquadDataset:
  """Load n' encode SQuAD data from a parsed JSON document.

  Args:
    squad_json: (obj) The SQuAD dataset, as returned by json.load(...).
    qencoder: The query encoder function, which accepts an np array of
      serialized tensorflow.Examples.
    rencoder: The response encoder function, which accepts an np array of
      serialized tensorflow.Examples.

  Returns:
    A SquadDataset instance.
  """
  qa_count = 0
  queries = [] # type: List[QuestionAnswerInput]
  master_dict = {}  # type: Dict[Tuple[str, str], np.ndarray]
  questions = set()  # type: Set[str]

  master_index = []  # type: List[IndexedItem]
  seen_responses = set()  # type: Set[Tuple[str, str]]
  for question, answer, document, paragraph in generate_examples(squad_json):
    questions.add(question)
    queries.append(
        QuestionAnswerInput(question, answer, document, paragraph.id, None))
    qa_count += 1

    for sentence in paragraph.sentences:
      if (sentence, document) not in seen_responses:
        seen_responses.add((sentence, document))
        master_index.append(IndexedItem(sentence, document, None, paragraph.id))
  logging.info("questions=%s, QA inputs=%s, index_size=%s",
               len(questions), qa_count, len(master_index))

  for i, chunk in enumerate(_chunks(queries, 100)):
    qencs = qencoder(chunk)
    for j in range(qencs.shape[0]):
      queries[i * 100 + j].encoding = qencs[j]
    if i % 10 == 0:
      logging.info("questions: encoded %s of %s...", i * 100, len(queries))

  for i, chunk in enumerate(_chunks(master_index, 50)):
    rencs = rencoder(chunk)
    for j in range(rencs.shape[0]):
      master_index[i * 50 + j].encoding = rencs[j]
    if i % 20 == 0:
      logging.info("answers: encoded %s of %s...", i * 50, len(master_index))

  # Given a sentence, stores its index or indices within the master_index.
  response_index = collections.defaultdict(list)  # type: Dict[str, List[int]]
  for i, (sentence, _, _, _) in enumerate(master_index):
    response_index[sentence].append(i)

  return SquadDataset(queries, master_index, response_index)


def generate_examples(data: Any) \
      -> Generator[Tuple[str, str, str, List[str]], None, None]:
  """Generates SQuAD examples.

  Args:
    data: (object) A python object returned by json.load(...)

  Yields:
    - the query.
    - the response sentence.
    - the response context.
    - a Paragraph instance.
  """

  # Loop through the SQuAD JSON file and perform the conversion. The basic
  # outline, which is mirrored in the for-loops below, is as follows:
  #
  # data ---< Passage ---< Paragraph ---< Question ---< Answer
  #
  # The loops below convert every answer into a QuestionAnswerInput.

  counts = [0, 0, 0]
  for passage in data["data"]:
    counts[0] += 1
    for paragraph in passage["paragraphs"]:
      counts[1] += 1

      paragraph_text = paragraph["context"]
      sentence_breaks = list(sb_sed.infer_sentence_breaks(paragraph_text))
      para = Paragraph(counts[1],
                       [paragraph_text[start:end]
                        for (start, end) in sentence_breaks])
      for qas in paragraph["qas"]:
        # The answer sentences that have been output for the current question.
        answer_sentences = set()  # type: Set[str]

        counts[2] += 1
        for answer in qas["answers"]:
          answer_start = answer["answer_start"]
          # Map the answer fragment back to its enclosing sentence.
          sentence = None
          for start, end in sentence_breaks:
            if start <= answer_start < end:
              sentence = paragraph_text[start:end]
              break

          # Avoid generating duplicate answer sentences.
          if sentence not in answer_sentences:
            answer_sentences.add(str(sentence))
            yield (qas["question"], str(sentence), paragraph_text, para)

  logging.info("processed %s passages, %s paragraphs, and %s questions.",
               *counts)


# Ref: https://stackoverflow.com/questions/312443/
def _chunks(l: Sequence, n: int = 5) -> Generator[Sequence, None, None]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == "__main__":
  main()
