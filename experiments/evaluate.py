from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    labelled_embeddings = active_learner.classifier.embed(train)
    test_embeddings = active_learner.classifier.embed(test)

    r = {
        "Train accuracy": accuracy_score(y_pred, train.y),
        "Test accuracy": accuracy_score(y_pred_test, test.y),
        "Train F1": f1_score(y_pred, train.y),
        "Test F1": f1_score(y_pred_test, test.y),
        "Train precision": precision_score(y_pred, train.y),
        "Test precision": precision_score(y_pred_test, test.y),
        "Train recall": recall_score(y_pred, train.y),
        "Test recall": recall_score(y_pred_test, test.y),
        "Test predictions": y_pred_test,
        "Test ground truth": test.y,
        "Test embeddings": test_embeddings,
        "Labelled data embeddings": labelled_embeddings,
        "Labelled data labels": train.y,
    }

    print("Test accuracy:", r["Test accuracy"], "Test F1:", r["Test F1"])
    return r
