import { app } from "../../scripts/app.js";


// In the main function where widgetLogic is called
function widgetLogic(node, widget) {
    console.log(node);

    if (widget.name == "num_files") {
        console.log("Entered with value: " + widget.value);
        
        const numFiles = widget.value; // Número de archivos
        node.widgets = node.widgets || []; // Asegúrate de que `node.widgets` esté inicializado

        // Limpia cualquier archivo existente en la lista a partir de la segunda posición
        node.widgets = node.widgets.slice(0, 2);

        // Agrega nuevos widgets dinámicamente según `num_files`
        for (let i = 1; i <= numFiles; i++) {
            node.widgets.push({
                "type": "text",
                "name": `file_${i}`,
                "value": `ruta_del_csv_${i}.csv`,
                "options": {
                    "inputIsOptional": true
                },
                "last_y": 230 + (i - 1) * 30 // Ajusta la posición en Y para cada nuevo widget
            });
        }

        console.log("Updated widgets:", node.widgets);
    }
}

app.registerExtension({
    name: "PromptBasedTagRandomizer.widgethider",
    nodeCreated(node) {
        for (const w of node.widgets || []) {
            let widgetValue = w.value;

            // Store the original descriptor if it exists
            let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value');
            if (!originalDescriptor) {
                originalDescriptor = Object.getOwnPropertyDescriptor(w.constructor.prototype, 'value');
            }

            widgetLogic(node, w);

            Object.defineProperty(w, 'value', {
                get() {
                    // If there's an original getter, use it. Otherwise, return widgetValue.
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(w)
                        : widgetValue;

                    return valueToReturn;
                },
                set(newVal) {

                    // If there's an original setter, use it. Otherwise, set widgetValue.
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(w, newVal);
                    } else {
                        widgetValue = newVal;
                    }

                    widgetLogic(node, w);
                }
            });
        }
        setTimeout(() => {initialized = true;}, 500);
    }
})