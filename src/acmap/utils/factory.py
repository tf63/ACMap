from acmap.utils.context import Context


def get_model(context: Context):
    model_name = context.config.exp.name
    name = model_name.lower()
    if name == 'acmap':
        from acmap.models.acmap import Learner
    else:
        raise NotImplementedError(f'Model {model_name} is not implemented.')

    return Learner(context=context)
