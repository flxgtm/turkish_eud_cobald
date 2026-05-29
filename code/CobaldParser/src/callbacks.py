from transformers import TrainerCallback


class GradualUnfreezeCallback(TrainerCallback):
    """Unfreeze one encoder layer per epoch, deepest first."""

    def __init__(self, warmup: int = 1, interval: int = 3):
        self.warmup = warmup
        self.interval = interval

    def on_train_begin(self, args, state, control, model = None, **kwargs):
        # Freeze encoder at start
        for param in model.encoder.parameters():
            param.requires_grad = False

    def on_epoch_begin(self, args, state, control, model = None, **kwargs):
        epoch = int(state.epoch)

        # Keep encoder frozen during warmup
        if epoch < self.warmup:
            return
        
        layers = model.encoder.get_transformer_layers()
        top_layer_idx = len(layers) - 1
        last_frozen_layer_idx = top_layer_idx - epoch * self.interval

        # Gradually unfreeze layers from top to bottom or unfreeze encoder entirely
        # including the embeddings, if all layers are already unfrozen.
        if last_frozen_layer_idx < 0:
            for param in model.encoder.parameters():
                param.requires_grad = True
        else:
            for layer in layers[top_layer_idx:last_frozen_layer_idx:-1]:
                for param in layer.parameters():
                    param.requires_grad = True