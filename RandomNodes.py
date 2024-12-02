import random
import pandas as pd
import numpy as np
import torch  # Añadir esta importación al inicio del archivo
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

        # Define possible shapes
        shapes = ['circle', 'rectangle', 'triangle']
        shape = random.choice(shapes)
        
        # Calculate size for the shape (30% of the smallest dimension)
        size = min(width, height) * 0.3
        
        if shape == 'circle':
            circle = plt.Circle((pos_x, pos_y), size / 2, color=shape_color)
            ax.add_patch(circle)
            
        elif shape == 'rectangle':
            rect = Rectangle((pos_x - size / 2, pos_y - size / 2), size, size, color=shape_color)
            ax.add_patch(rect)
            
        else:  # triangle
            points = [
                (pos_x, pos_y + size / 2),  # top
                (pos_x - size / 2, pos_y - size / 2),  # bottom left
                (pos_x + size / 2, pos_y - size / 2)   # bottom right
            ]
            triangle = plt.Polygon(points, color=shape_color)
            ax.add_patch(triangle)

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

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "Randomizer"

    @staticmethod
    def execute(width, height, color1, color2, color3, color4):
        colors = [color1, color2, color3, color4]
        print(f"Usando colores hexadecimales: {colors}")

        # Validar ancho y alto
        if width <= 0 or height <= 0:
            raise ValueError("El ancho y el alto deben ser mayores que cero.")

        # Crear la figura y el eje
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.axis('off')

        # Calcular el ancho de cada rectángulo
        rect_width = width // 4  # Dividir en 4 columnas
        rect_height = height
        for i, color in enumerate(colors):
            # Cambiar las coordenadas para disposición horizontal
            x = i * rect_width
            rect = Rectangle((x, 0), rect_width, rect_height, color=color)
            ax.add_patch(rect)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Ajustar el tamaño de la figura
        
        # Convertir la figura a imagen
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Convertir a tensor
        img_array = img_array.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(img_array)

        # Convertir a tensor latente
        latent_tensor = tensor.unsqueeze(0)  # Añadir dimensión de batch

        # Asegurarse de que el tensor latente tenga la forma correcta (batch_size, channels, height, width)
        if latent_tensor.dim() == 3:
            latent_tensor = latent_tensor.unsqueeze(0)

        # Asegurarse de que el tensor latente tenga 4 dimensiones
        if latent_tensor.dim() != 4:
            raise ValueError("El tensor latente debe tener 4 dimensiones (batch_size, channels, height, width)")

        return (latent_tensor,)

# Exportar el nodo para que sea reconocido por ComfyUI.
NODE_CLASS_MAPPINGS = {
    "RandomTagSelector": RandomTagSelector,
    "RandomColorNode": RandomColorNode,
    "RandomGeometricShapeNode": RandomGeometricShapeNode,
    "ColorPaletteNode": ColorPaletteNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomTagSelector": "Random Tag Selector",
    "RandomColorNode": "Random Color Node",
    "RandomGeometricShapeNode": "Random Geometric Shape",
    "ColorPaletteNode": "Color Palette Generator"
}
