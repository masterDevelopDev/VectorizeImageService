import json
import torch
import numpy as np
from PIL import Image
import clip
import io
import base64
import io

# Check if CUDA is available and set the appropriate device (GPU or CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load('ViT-B/32', device=device)
model.eval()


def model_fn(model_dir):
    """
    Placeholder function for environments expecting a model loading function.

    Since the model is loaded globally, this function does not need to load the model again.
    It exists to comply with interfaces expecting a model loading function and can be used
    to perform additional setup if required.

    Returns:
    - dict: A dictionary containing the globally loaded model and preprocess.
    """
    return {"model": model, "preprocess": preprocess}


def input_fn(request_body, request_content_type):
    """
    Preprocess the input data for the model.

    This function decodes the request body, extracts and preprocesses images
    from the base64 encoded strings.

    Args:
    - request_body (bytes): The request body containing the base64 encoded images.
    - request_content_type (str): The content type of the request.

    Returns:
    - torch.Tensor: A tensor of preprocessed images ready for model inference.
    """
    print("Preprocessing input data...")
    print(f'The request body is: {request_body}')
    data_str = request_body.decode('utf-8')
    # Parse the string as a JSON
    data_dict = json.loads(data_str)
    print(f'This what data_dict looks like: {data_dict}')
    base64_files = data_dict['Base64Files']
    print(f"Getting the preprocess model ...")
    list_preprocessed_images = []
    for base64_file in base64_files:
        print("Opening the image ...")
        image = Image.open(io.BytesIO(base64.b64decode(base64_file))).convert('RGB')
        print("Image has been opened.")
        list_preprocessed_images.append(preprocess(image))
    preprocessed_images_input = torch.tensor(np.stack(list_preprocessed_images))
    preprocessed_images_input = preprocessed_images_input.to(device)
    return preprocessed_images_input


def predict_fn(input_data, model_artifacts):
    """
    Generate predictions using the loaded model.

    This function takes the preprocessed input data and the loaded model and
    performs inference to generate embeddings of the images.

    Args:
    - input_data (torch.Tensor): The preprocessed image data.
    - model_artifacts (dict): A dictionary containing the loaded model.

    Returns:
    - dict: A dictionary containing the embeddings of the input images.
    """
    print(f"This is what the input data looks like: {input_data}")
    print("Generating prediction ...")
    print("The model has been loaded.")
    with torch.no_grad():
        print(f"Computing the embeddings ...")
        images_features = model.encode_image(input_data).float()
        print(f"The embeddings have been computed.")
    image_features_list = [images_features[idx].tolist() for idx in range(len(images_features))]
    print(f"The features list looks like this: {image_features_list}")
    return {'Embeddings': image_features_list}


# Function to serialize the prediction output
def output_fn(prediction_output, accept='application/x-npy'):
    """
    Serialize the prediction output.

    This function serializes the prediction output into the format specified
    by the 'accept' argument.

    Args:
    - prediction_output (dict): The output from the predict function.
    - accept (str): The MIME type to which the output should be serialized.

    Returns:
    - bytes or str: Serialized prediction output in the specified format.
    """
    print("Serializing the generated output.")
    print(f"The accept looks like this: {accept}")
    print(f"This is what the prediction output looks like: {prediction_output}")
    if accept == 'application/x-npy':
        output = np.array(prediction_output)
        buffer = io.BytesIO()
        np.save(buffer, output)
        buffer.seek(0)
        return buffer.getvalue()
    elif accept == 'application/json':
        return json.dumps(prediction_output)
    else:
        raise ValueError(f'Unsupported accept type: {accept}')
