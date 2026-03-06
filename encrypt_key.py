from cryptography.fernet import Fernet

# 1. Tạo hoặc sử dụng lại Encryption Key (Khóa này phải trùng với khóa trong secrets)
# Nếu anh đã có encryption_key trong secrets, hãy dán nó vào đây thay vì generate mới
encryption_key = Fernet.generate_key() 
cipher_suite = Fernet(encryption_key)

# 2. Dán Key Gemini của anh vào đây để mã hóa
gemini_key_to_encrypt = "AIzaSyCwMG94O5abc326iHv8GOYRK9YXxTdoUdw"

# Thực hiện mã hóa
encrypted_gemini_key = cipher_suite.encrypt(gemini_key_to_encrypt.encode())

print("-" * 50)
print("BƯỚC 1: Lưu dòng này vào encryption_key_for_decryption (nếu chưa có):")
print(encryption_key.decode())
print("\nBƯỚC 2: Lưu dòng này vào encrypted_gemini_api_key trong secrets:")
print(encrypted_gemini_key.decode())
print("-" * 50)