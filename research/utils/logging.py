from ignite.engine import Engine

# FIXME: Use logging module, not print statements
def log_training_loss(engine: Engine) -> None:
    print("Epoch[{}] Loss: {:.2f}".format(engine.state.epoch, engine.state.output))


# FIXME: Use logging module, not print statements
def log_training_results(engine: Engine) -> None:
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
            engine.state.epoch, metrics["accuracy"], metrics["nll"]
        )
    )


# FIXME: Use logging module, not print statements
def log_validation_results(engine: Engine) -> None:
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
            engine.state.epoch, metrics["accuracy"], metrics["nll"]
        )
    )


# TODO: Add Visdom logging callbacks
# TODO: Add Slack notification callback
