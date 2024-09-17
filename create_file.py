import base64


def encode_env_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                encoded_value = base64.b64encode(
                    value.encode('utf-8')).decode('utf-8')
                f_out.write(f"{key}={encoded_value}\n")
            else:
                f_out.write(line + '\n')


# Usage
input_file = '.env_prod'
output_file = '.env.base64'
encode_env_file(input_file, output_file)
print(f"Encoded secrets have been written to {output_file}")
