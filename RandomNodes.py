import torch  # Añadir esta importación al inicio del archivo
import random
import pandas as pd
import numpy as np
import folder_paths
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

class RandomColorNode:
    last_seed = 0

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                    "step": 1,
                    "display": "number",
                }),
                "seed_control": (["fixed", "randomized", "increment", "decrement"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Randomizer"

    @staticmethod
    def execute(seed, seed_control):
        if seed_control == "fixed":
            current_seed = seed
        elif seed_control == "randomized":
            current_seed = random.randint(0, 0xffffffff)
        elif seed_control == "increment":
            current_seed = (RandomColorNode.last_seed + 1) % 0xffffffff
        else:  # decrement
            current_seed = (RandomColorNode.last_seed - 1) % 0xffffffff

        RandomColorNode.last_seed = current_seed
        random.seed(current_seed)
        # Generate random RGB values
        r = random.random()
        g = random.random()
        b = random.random()

        # Convert to hex format
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

        print(f"Random color generated with seed {current_seed}: {hex_color}")
        return (hex_color,)



class RandomTagSelector:
    """
    Nodo para cargar múltiples archivos CSV, seleccionar aleatoriamente un tag por categoría
    y devolver un string con los resultados combinados.
    """
    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_files": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "lazy": False
                }),
                "seed": ("INT", {
                    "default": 42,  # Valor por defecto para la semilla
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number",
                    "lazy": False
                }),
            },
            "optional": {
                "file_1": ("STRING", {"default": "ruta_del_csv_1.csv", "lazy": True}),
                "file_2": ("STRING", {"default": "ruta_del_csv_2.csv", "lazy": True}),
                "file_3": ("STRING", {"default": "ruta_del_csv_3.csv", "lazy": True}),
                # Agregar más archivos si es necesario.
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("random_tags",)
    FUNCTION = "execute"
    CATEGORY = "Randomizer"

    @staticmethod
    def execute(num_files,seed, **files):
        """
        Método principal para procesar los archivos CSV y generar la salida.
        """
        combined_tags = []

        random.seed(seed)

        # Cargar cada archivo y seleccionar tags aleatorios por categoría.
        for i in range(1, num_files + 1):
            file_path = files.get(f"file_{i}")
            if not file_path:
                continue

            # Cargar archivo CSV y validar estructura.
            try:
                data = pd.read_csv(file_path)
                if "categoria" not in data.columns or "tag" not in data.columns:
                    raise ValueError(f"El archivo {file_path} no tiene las columnas requeridas: 'categoria' y 'tag'.")
            except Exception as e:
                print(f"Error al cargar el archivo {file_path}: {e}")
                continue

            # Agrupar por categoría y seleccionar un tag aleatorio para cada categoría.
            categories = data.groupby("categoria")
            for category, group in categories:
                # Verificar que hay tags en el grupo
                if group["tag"].notna().any():  # Verifica si hay tags no nulos
                    tag = random.choice(group["tag"].dropna().tolist())  # Eliminar valores nulos
                    print(f"Tag seleccionado para {category}: {tag}")
                    combined_tags.append(tag)
                else:
                    print(f"Advertencia: No hay tags disponibles para la categoría '{category}'")
                    combined_tags.append("No Tag Available")  # Si no hay tags, agrega un valor por defecto

        # Combinar tags seleccionados en una cadena de texto.
        return (",".join(combined_tags),)

class RandomGeometricShapeNode:
    last_seed = 0  # Add class variable to track last seed

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "pos_x": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "pos_y": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "background_color": ("STRING", {"default": "#FFFFFF"}),
                "shape_color": ("STRING", {"default": "#000000"}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                    "step": 1,
                    "display": "number"
                }),
                "seed_control": (["fixed", "randomized", "increment", "decrement"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Randomizer"

    @staticmethod
    def execute(width, height, pos_x, pos_y, background_color, shape_color, seed, seed_control):
        # Handle seed control
        if seed_control == "fixed":
            current_seed = seed
        elif seed_control == "randomized":
            current_seed = random.randint(0, 0xffffffff)
        elif seed_control == "increment":
            current_seed = (RandomGeometricShapeNode.last_seed + 1) % 0xffffffff
        else:  # decrement
            current_seed = (RandomGeometricShapeNode.last_seed - 1) % 0xffffffff

        RandomGeometricShapeNode.last_seed = current_seed
        random.seed(current_seed)
        
        print(f"Generating shape with seed {current_seed}")
        
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.axis('off')
        ax.set_facecolor(background_color)

        # Generate only a rectangle
        size = min(width, height) * 0.3
        rect = Rectangle((pos_x - size / 2, pos_y - size / 2), size, size, color=shape_color)
        ax.add_patch(rect)

        # Convertir imagen a tensor
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        img_array = img_array.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(img_array)
        return (tensor,)

class ColorPaletteNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "color1": ("STRING", {"default": "#FF0000"}),
                "color2": ("STRING", {"default": "#00FF00"}),
                "color3": ("STRING", {"default": "#0000FF"}),
                "color4": ("STRING", {"default": "#FFFF00"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Randomizer"

    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    @staticmethod
    def execute(width, height, color1, color2, color3, color4):
        rgb_color = ColorPaletteNode.hex_to_rgb(color1)  # Convert hex to RGB
        print(f"Using RGB color: {rgb_color}")

        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be greater than zero.")

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.axis('off')

        rect = Rectangle((0, 0), width, height, color=rgb_color)
        ax.add_patch(rect)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        img_array = img_array.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(img_array)
        return (tensor,)

class CSVPromptLoader:
    """
    Loads csv file with prompts. The CSV file should have two columns: 'positive prompt' and 'negative prompt'.
    """

    last_row = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_file": ("STRING", {"default": "/path/to/prompts.csv"}),
                "row_number": ("INT", {
                    "default": 1,
                    "min": 1,
                    "display": "number",
                }),
                "row_control": (["Increment", "Randomize"],),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive prompt", "negative prompt")
    FUNCTION = "execute"
    CATEGORY = "CSV Loaders"   

    @staticmethod
    def load_prompt_csv(prompt_path: str):
        """Loads csv file with prompts. The CSV file should have two columns: 'positive prompt' and 'negative prompt'.
        
        Returns:
            list: List of rows with prompts.
        """
        prompts = []
        if not os.path.exists(prompt_path):
            print(f"""Error. No prompts.csv found at {prompt_path}. Please check the path and try again.""")
            return prompts
        try:
            data = pd.read_csv(prompt_path)
            if "positive prompt" not in data.columns or "negative prompt" not in data.columns:
                raise ValueError("The CSV file must contain 'positive prompt' and 'negative prompt' columns.")
            prompts = data[["positive prompt", "negative prompt"]].iloc[1:].values.tolist()  # Ignore the first row
        except Exception as e:
            print(f"""Error loading prompts.csv. Please check the path and try again.
                    Error: {e}
            """)
        return prompts

    def execute(self, csv_file, row_number, row_control):
        prompt_csv = self.load_prompt_csv(csv_file)
        if not prompt_csv:
            return ("", "")

        row_number = row_number - 1

        if row_control == "Increment":
            row_index = (CSVPromptLoader.last_row + 1) % len(prompt_csv)
        elif row_control == "Randomize":
            row_index = random.randint(0, len(prompt_csv) - 1)

        selected_row = prompt_csv[row_number]
        row_number = row_index
        
        positive_prompt = selected_row[0]
        negative_prompt = selected_row[1]

        return (positive_prompt, negative_prompt)


# Exportar el nodo para que sea reconocido por ComfyUI.
NODE_CLASS_MAPPINGS = {
    "RandomTagSelector": RandomTagSelector,
    "RandomColorNode": RandomColorNode,
    "RandomGeometricShapeNode": RandomGeometricShapeNode,
    "ColorPaletteNode": ColorPaletteNode,
    "CSVPromptLoader": CSVPromptLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomTagSelector": "Random Tag Selector",
    "RandomColorNode": "Random Color Node",
    "RandomGeometricShapeNode": "Random Geometric Shape",
    "ColorPaletteNode": "Color Palette Generator",
    "CSVPromptLoader": "CSV Prompt Generator"

}
