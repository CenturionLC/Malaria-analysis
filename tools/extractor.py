import os
import rarfile


input_folder = 'datasets'

def extractFiles(input_folder, output_folder, files):
    os.makedirs(output_folder, exist_ok=True)

    for rar_filename in files:
        rar_path = os.path.join(input_folder, rar_filename)
        
        try:
            with rarfile.RarFile(rar_path) as rf:
                print(f"Contents of {rar_filename}:")
                for file_info in rf.infolist():
                    print(f" - {file_info.filename} ({file_info.file_size} bytes)")
                
                rf.extractall(output_folder)
            print(f"Successfully extracted {rar_filename}")
        except rarfile.NeedFirstVolume:
            print(f"Error: {rar_filename} is not the first volume.")
        except Exception as e:
            print(f"Error extracting {rar_filename}: {e}")

    print("Extraction complete.")

def main():
    output_folder = 'temporary/ghana'
    
    ghana_files = [
        'Thick_Ghana.part1.rar',
        'Thin_Images_Ghana.rar',
    ]

    extractFiles(input_folder, output_folder, ghana_files)
    print("All Ghana files extracted successfully.")


    uganda_files = [
        'Thin_Uganda.rar',
    ]

    output_folder = 'temporary/uganda'

    extractFiles(input_folder, output_folder, uganda_files)
    print("All Uganda files extracted successfully.")

if __name__ == "__main__":
    main()

