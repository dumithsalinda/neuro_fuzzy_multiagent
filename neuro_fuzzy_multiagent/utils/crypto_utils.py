from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.exceptions import InvalidSignature
import base64


# Generate a new RSA key pair (for testing/demo)
def generate_key_pair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


# Sign data with private key
def sign_data(data: bytes, private_key) -> str:
    signature = private_key.sign(
        data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


# Verify signature with public key
def verify_signature(data: bytes, signature_b64: str, public_key) -> bool:
    signature = base64.b64decode(signature_b64)
    try:
        public_key.verify(
            signature,
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )
        return True
    except InvalidSignature:
        return False


# Load public key from PEM file
def load_public_key(pem_path: str):
    with open(pem_path, "rb") as f:
        pem_data = f.read()
    return serialization.load_pem_public_key(pem_data)


# Load private key from PEM file
def load_private_key(pem_path: str, password: str = None):
    with open(pem_path, "rb") as f:
        pem_data = f.read()
    return serialization.load_pem_private_key(
        pem_data, password=password.encode() if password else None
    )


# Save key to PEM file
def save_key_to_pem(key, pem_path: str, private: bool = False, password: str = None):
    with open(pem_path, "wb") as f:
        if private:
            enc_alg = (
                serialization.BestAvailableEncryption(password.encode())
                if password
                else serialization.NoEncryption()
            )
            f.write(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=enc_alg,
                )
            )
        else:
            f.write(
                key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
