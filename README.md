# Detecta, localiza, clasifica: visión por computadora en acción

Código del taller. Son cuatro proyectos que se controlan con las manos frente a la cámara web,
hechos con Python, OpenCV y MediaPipe. Funcionan en Windows, macOS y Linux.

Esta guía tiene tres partes: preparar tu computadora (una sola vez), ejecutar los proyectos y
qué hacer si algo falla.

---

## Parte 1. Preparar tu computadora (una sola vez)

Necesitas una computadora con cámara web y conexión a internet para la instalación.

### Paso 1. Revisa si ya tienes Python

Abre una terminal (en Windows, busca **PowerShell**; en macOS, **Terminal**) y escribe:

| Sistema | Comando |
|---|---|
| Windows | `python --version` |
| macOS y Linux | `python3 --version` |

Si responde con una versión entre **3.10 y 3.13**, pasa al paso 2.

Si marca error, abre la tienda de Microsoft o muestra otra versión, instala Python:

- **Windows**: descárgalo de <https://www.python.org/downloads/>. Al iniciar el instalador,
  marca la casilla **«Add python.exe to PATH»** antes de presionar *Install Now*. Al terminar,
  cierra PowerShell y vuelve a abrirlo.
- **macOS**: descarga el instalador de <https://www.python.org/downloads/> y sigue los pasos.
- **Linux (Ubuntu o Debian)**: `sudo apt install python3 python3-venv python3-pip`

### Paso 2. Descarga el código

Con Git:

```bash
git clone https://github.com/USUARIO/REPOSITORIO.git
cd REPOSITORIO
```

Sin Git: en la página del repositorio presiona el botón verde **Code**, elige **Download ZIP**,
descomprime el archivo y abre una terminal dentro de esa carpeta.

Todos los comandos que siguen se escriben **dentro de la carpeta del proyecto**, la que contiene
el archivo `requirements.txt`.

### Paso 3. Crea el entorno virtual

El entorno virtual es una carpeta llamada `.venv` donde se instalan las bibliotecas del taller,
separadas del resto de tu computadora. Viene incluido con Python; no hay que instalar nada más.

| Sistema | Comando |
|---|---|
| Windows | `python -m venv .venv` |
| macOS y Linux | `python3 -m venv .venv` |

### Paso 4. Activa el entorno virtual

| Sistema | Comando |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (CMD) | `.venv\Scripts\activate.bat` |
| macOS y Linux | `source .venv/bin/activate` |

Sabrás que está activo porque al inicio de la línea aparece **`(.venv)`**.

Si PowerShell responde que «la ejecución de scripts está deshabilitada», escribe este comando,
acepta con `S` y vuelve a activar:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Paso 5. Instala las bibliotecas

Con el entorno activo:

```bash
pip install -r requirements.txt
```

Tarda unos minutos. Solo se hace una vez.

### ¿Usas Anaconda o Miniconda?

Puedes sustituir los pasos 3, 4 y 5 por estos comandos:

```bash
conda create -n taller python=3.12 -y
conda activate taller
pip install -r requirements.txt
```

### Cuando regreses otro día

No hay que instalar de nuevo. Abre la terminal en la carpeta del proyecto y solo **activa el
entorno** (paso 4). Para salir del entorno escribe `deactivate`.

---

## Parte 2. Ejecutar los proyectos

Antes de ejecutar cualquiera, revisa que el entorno esté activo: debe verse `(.venv)`.
En macOS y Linux, si `python` no funciona, escribe `python3`.

La primera vez que ejecutes un proyecto se descargará un archivo de unos 7 MB (el modelo que
detecta las manos), así que necesitas internet. Después ya no hace falta.

Todos los proyectos se cierran con la tecla **`q`** o **`Esc`**, con la ventana del programa
seleccionada.

### Calentamiento: ver los puntos de la mano

```bash
python pasos/paso1_manos.py
```

Muestra los 21 puntos que el programa encuentra en tu mano. Sirve para comprobar que la cámara
y la instalación funcionan.

### Proyecto 1. Pintar en el aire

```bash
python pizarra.py
```

| Gesto | Qué hace |
|---|---|
| Solo el índice arriba | Dibuja |
| Índice y medio arriba | Selecciona: deja los dedos medio segundo sobre un botón de la barra |
| Palma abierta o puño | Mueve la mano sin dibujar |

Teclas: `1` a `8` color · `g` goma · `z` deshacer · `c` borrar todo · `s` guardar el dibujo ·
`p` modo pizarrón.

### Proyecto 2. Fruit Ninja

```bash
python fruit_ninja.py
```

Para empezar, pasa el dedo índice sobre la sandía que dice **INICIO**. Corta las frutas con el
índice y evita las bombas. Tienes tres vidas. Tecla `r` para reiniciar.

Si corta muy poco o demasiado, cambia la sensibilidad (el valor normal es 0.25; un número menor
corta más fácil):

```bash
python fruit_ninja.py --corte 0.15
```

### Proyecto 3. Cubo Rubik con gestos

```bash
python cubo_rubik.py
```

El cubo inicia desordenado y el reto es ordenarlo.

| Mano | Gesto | Qué hace |
|---|---|---|
| Izquierda, fuera del recuadro | Palma abierta y mover | Gira el cubo completo |
| Izquierda, fuera del recuadro | Puño | Pausa, para acomodar la mano |
| Derecha, dentro del recuadro | Pulgar e índice cerca | Apunta a un cuadrito del cubo |
| Derecha, dentro del recuadro | Pellizcar y arrastrar | Gira esa capa |

Teclas: `p` pista · `m` mezclar de nuevo · `z` deshacer · `r` dejarlo ordenado ·
`espacio` vista inicial · `a` mostrar u ocultar la ayuda.

Con la tecla `p` aparece una flecha amarilla: pellizca ese cuadrito y arrástralo como indica la
flecha. Siguiendo las pistas una tras otra, el cubo queda ordenado.

Otras formas de iniciarlo:

```bash
python cubo_rubik.py --pistas          # inicia con las pistas encendidas
python cubo_rubik.py --movimientos 4   # mezcla corta, más fácil
python cubo_rubik.py --ordenado        # inicia ordenado, para practicar los gestos
```

¿Quieres probar el cubo sin cámara, con el ratón y el teclado?

```bash
python pasos/paso2_cubo_mouse.py
```

### Proyecto 4. Abecedario de señas

Este proyecto tiene tres programas que se ejecutan **en este orden**:

```bash
python senas/grabar.py      # 1. Grabar ejemplos de cada seña
python senas/entrenar.py    # 2. Entrenar el modelo con esos ejemplos
python senas/reconocer.py   # 3. Reconocer las señas en vivo
```

1. **Grabar.** Haz una seña con una mano y, sin deshacerla, presiona con la otra la tecla de su
   letra (por ejemplo `a`). Tras una cuenta regresiva graba unos 4 segundos; mientras tanto,
   mueve un poco la mano. Graba al menos dos señas distintas, dos veces cada una.
   La tecla `-` borra la última grabación.
2. **Entrenar.** No abre ventana: escribe en la terminal qué tan bien aprendió.
3. **Reconocer.** Haz una de las señas que grabaste. Si la sostienes casi un segundo, la letra
   se escribe en pantalla. Teclas: `espacio` agrega un espacio · `retroceso` borra una letra ·
   `c` borra todo.

No hace falta saber lengua de señas: el programa aprende las formas de mano que tú le enseñes.
Para empezar, prueba con tres gestos fáciles de distinguir: puño cerrado, mano abierta y
«amor y paz».

---

## Parte 3. Si algo no funciona

| Qué pasa | Qué hacer |
|---|---|
| `python` no se reconoce como comando | Instala Python (paso 1) y abre de nuevo la terminal. En macOS y Linux usa `python3` |
| `No module named cv2` o `mediapipe` | El entorno no está activo. Actívalo (paso 4) y, si hace falta, repite el paso 5 |
| PowerShell no deja activar el entorno | Ejecuta `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a activar |
| «No se pudo abrir la cámara» | Cierra Zoom, Teams o Meet. Si tienes más de una cámara, agrega `--camara 1` al comando |
| En macOS no se ve la cámara | Configuración del Sistema → Privacidad y seguridad → Cámara → activa Terminal y vuelve a abrirla |
| El programa va lento | Agrega `--ancho 960 --alto 540` al comando y conecta la laptop a la corriente |
| No detecta la mano | Busca luz de frente y que la mano completa se vea en la imagen |
| Señas dice «No hay modelo» | Primero ejecuta `senas/grabar.py` y después `senas/entrenar.py` |
