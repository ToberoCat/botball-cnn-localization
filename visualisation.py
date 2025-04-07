import visualkeras

from src.model_builder import build_better_model

model = build_better_model()

def text_callable(layer_index, layer):
    above = bool(layer_index % 2)

    output_shape = [x for x in list(layer.output.shape) if x is not None]

    if isinstance(output_shape[0], tuple):
        output_shape = list(output_shape[0])
        output_shape = [x for x in output_shape if x is not None]

    output_shape_txt = ""

    for ii in range(len(output_shape)):
        output_shape_txt += str(output_shape[ii])
        if ii < len(output_shape) - 2:
            output_shape_txt += "x"
        if ii == len(output_shape) - 2:
            output_shape_txt += "\n"

    activation = layer.activation._api_export_path.split(".")[-1] if hasattr(layer, 'activation') else None
    if activation:
        output_shape_txt += f"\n{activation}"
    return output_shape_txt, above


visualkeras.layered_view(model,
                         legend=True,
                         text_callable=text_callable,
                         shade_step=15,
                         scale_z=0.5,
                         scale_xy=2,
                         padding=20,
                         to_file='model.png',
                         draw_funnel=True).show()