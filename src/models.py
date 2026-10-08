"""Keras models. Every model: 0-255 RGB input -> (scaling layers) -> 'backbone' -> GAP -> Dropout -> 1 logit.
Scaling is INSIDE the model, so training, Grad-CAM and the web apps can never disagree about preprocessing.
(ResNet-18 does not exist in keras.applications, so the ResNet option is ResNet50V2.)"""
import keras
from keras import layers

from .config import IMG_SIZE, NO_PRETRAINED


def _pre(name):
    if name == "efficientnet_b0":            # Keras EfficientNet rescales internally (expects 0-255)
        return []
    if name in ("mobilenet_v2", "resnet50v2"):
        return [layers.Rescaling(1 / 127.5, offset=-1, name="pre_rescale")]
    if name == "densenet121":
        return [layers.Rescaling(1 / 255.0, name="pre_rescale"),
                layers.Normalization(mean=[0.485, 0.456, 0.406],
                                     variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2], name="pre_norm")]
    if name == "custom_cnn":
        return [layers.Rescaling(1 / 255.0, name="pre_rescale")]
    raise ValueError(name)


def _custom_backbone():
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = inp
    for f in (32, 64, 128, 256, 256):
        for _ in range(2):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization(momentum=0.9)(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    return keras.Model(inp, x)


def _pretrained_backbone(name, pretrained):
    w = "imagenet" if (pretrained and not NO_PRETRAINED) else None
    kw = dict(include_top=False, weights=w, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    app = keras.applications
    return {"mobilenet_v2": app.MobileNetV2, "efficientnet_b0": app.EfficientNetB0,
            "resnet50v2": app.ResNet50V2, "densenet121": app.DenseNet121}[name](**kw)


def build_model(name: str, pretrained: bool = True, dropout: float = 0.3) -> keras.Model:
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x = inp
    for l in _pre(name):
        x = l(x)
    base = _custom_backbone() if name == "custom_cnn" else _pretrained_backbone(name, pretrained)
    base = keras.Model(base.input, base.output, name="backbone")
    x = base(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    out = layers.Dense(1, name="logit")(x)                 # raw logit; sigmoid is applied outside
    return keras.Model(inp, out, name=name)


def _walk(layer):
    yield layer
    for sub in getattr(layer, "layers", None) or []:
        yield from _walk(sub)


def set_backbone_trainable(model, flag: bool, keep_bn_frozen: bool = True):
    """Freeze/unfreeze the backbone. When fine-tuning a pretrained net, BatchNorm stays frozen
    (standard practice: small chest-X-ray batches would otherwise wreck the pretrained statistics)."""
    bb = model.get_layer("backbone")
    bb.trainable = flag
    if flag and keep_bn_frozen:
        for l in _walk(bb):
            if isinstance(l, layers.BatchNormalization):
                l.trainable = False
