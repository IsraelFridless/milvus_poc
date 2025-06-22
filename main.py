import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import sys
import os


def validate_input():
    if len(sys.argv) != 2 or not os.path.exists(sys.argv[1]):
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    return sys.argv[1]


def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def generate_embedding(image_tensor):
    model = torch.nn.Sequential(*list(models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT).children())[:-1])
    model.eval()
    with torch.no_grad():
        return model(image_tensor).squeeze().numpy()


def connect_to_milvus():
    try:
        connections.connect(host='localhost', port='19530')
    except Exception:
        print("Failed to connect to Milvus")
        sys.exit(1)


def prepare_collection(collection_name):
    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=2048),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500)
        ]
        schema = CollectionSchema(fields=fields)
        collection = Collection(name=collection_name, schema=schema)

        # Create index only after inserting schema
        collection.create_index("embedding", {
            "metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
    else:
        collection = Collection(name=collection_name)
    return collection



def insert_embedding(collection, embedding, image_path):
    collection.insert([[embedding.tolist()], [image_path]])
    collection.flush()


def main():
    image_path = validate_input()
    image_tensor = load_and_preprocess_image(image_path)
    embedding = generate_embedding(image_tensor)

    connect_to_milvus()
    collection = prepare_collection("images")
    insert_embedding(collection, embedding, image_path)

    print(f"Inserted embedding for {image_path}")


if __name__ == "__main__":
    main()
