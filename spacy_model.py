import spacy
from spacy.util import minibatch, compounding
from tqdm import tqdm
from pathlib import Path
def pre_process_data(train, test):
    """Load data from the IMDB dataset."""

    text_train = tuple(train['text'].values)
    cats_train = tuple([{'POSITIVE': bool(i)} for i in train['label'].values])

    text_eval = tuple(test['text'].values)
    cats_eval = tuple([{'POSITIVE': bool(i)} for i in test['label'].values])

    return (text_train, cats_train), (text_eval, cats_eval)

def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0   # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0   # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_a': accuracy, 'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}

def spacy_train(train, test, model=None, output_dir=None, n_iter=3):
    model_stats = []
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')

    # add label to text classifier
    textcat.add_label('POSITIVE')

    # load the IMDB dataset
    (train_texts, train_cats), (dev_texts, dev_cats) = pre_process_data(train, test)
    print("Using {} examples ({} training, {} evaluation)"
          .format(len(train_texts) + len(dev_texts), len(train_texts), len(dev_texts)))
    train_data = list(zip(train_texts,
                          [{'cats': cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        for i in tqdm(range(n_iter)):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
                model_stats.append([i, losses['textcat'], scores['textcat_a'], scores['textcat_p'], scores['textcat_r'], scores['textcat_f']])

    # print a simple table
    print('{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('ITER', 'LOSS', 'A', 'P', 'R', 'F'))                                    
    for stat in model_stats:
        print('{0:.3f}\t{0:.3f}\t{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'.format(*stat))
                                    

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    return nlp