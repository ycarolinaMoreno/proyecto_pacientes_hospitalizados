# importamos las herramientas necesarias para
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd # Pandas nos ayuda a manejar datos en tablasm como si fuera un Excel.
import nltk # NLTK es una librería para procesar texto y analizar palabras. 
from nltk.tokenize import word_tokenize # Se usa para dividir un texto en palabras individuales.
from nltk.corpus import wordnet # Nos ayuda a encontrar sinonimos de palabras. 

# Indicamos la ruta donde NLTK buscará los datos descargados en nuestro computador. 
#nltk.data.path.append('C:\\Users\\latat\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\nltk')
import os
from nltk.tokenize import RegexpTokenizer #importamos la clase RegexpTokenizer de la librería nltk.tokenize para tokenizar palabras
nltk_path = os.path.join(os.getenv("APPDATA"), "nltk_data")
nltk.data.path.append(nltk_path)

# Descargamos las herramientas necesarias de NLTK para el análisis de palabras.

nltk.download('punkt') # Paquete para dividir frases en palabras.
nltk.download('wordnet') # Paquete para encontrar sinonimos de palabras en inglés.

# Función para cargar las los pacientes hospitalizados desde un archivo CSV

def load_pacientes():
    # Leemos el archivo que contiene información de los pacientes hospitalizados y seleccionamos las columnas más importantes
    df = pd.read_csv("Dataset/Dataset_Pacientes_LOS.csv", delimiter=";", quotechar='"', on_bad_lines="skip")[['id','entrance_date', 'discharge_date', 'Gender', 'Age', 'Disease','Service','LOS']]
    
    # Renombramos las columnas para que sean más faciles de entender
    df.columns = ['id', 'entrance_date', 'discharge_date', 'Gender', 'Age', 'Disease','service','LOS']
    
    # Llenamos los espacios vacíos con texto vacío y convertimos los datos en una lista de diccionarios 
    return df.fillna('').to_dict(orient='records')

# Cargamos los pacientes hospitalizados al iniciar la API para no leer el archivo cada vez que alguien pregunte por ellas.
pacientes_list = load_pacientes()

# Función para encontrar sinónimos de una palabra

def get_synonyms(word): 
    # Usamos WordNet para obtener distintas palabras que significan lo mismo.
    return{lemma.name().lower() for syn in wordnet.synsets(word) for lemma in syn.lemmas()}

# Creamos la aplicación FastAPI, que será el motor de nuestra API
# Esto inicializa la API con un nombre y una versión
app = FastAPI(title="Mi aplicación de pacientes hospitalizados", version="1.0.0")

# Ruta de inicio: Cuando alguien entra a la API sin especificar nada, verá un mensaje de bienvenida.

@app.get('/', tags=['Home'])
def home():
# Cuando entremos en el navegador a http://127.0.0.1:8000/ veremos un mensaje de bienvenida
    return HTMLResponse('<h1>Bienvenido a la API de pacientes hospitalizados</h1>')

# Obteniendo la lista de pacientes
# Creamos una ruta para obtener todos los pacientes hospitalizados
# Ruta para obtener todos los pacientes hospitalizados

@app.get('/pacientes', tags=['pacientes'])
def get_pacientes():
    # Si hay pacientes hospitalizados, los enviamos, si no, mostramos un error
    return pacientes_list or HTTPException(status_code=500, detail="No hay datos de pacientes hospitalizados")


# Ruta para obtener una película específica según su ID
@app.get('/pacientes/{id}', tags=['pacientes'])
def get_pacientes(id: str):
    # Buscamos en la lista de pacientes hospitalizados la que tenga el mismo ID
    return next((m for m in pacientes_list if m['id'] == id), {"detalle": "paciente no encontrado"})

# Ruta del chatbot que responde con pacientes según palabras clave del servicio


@app.get('/chatbot', tags=['Chatbot'])
def chatbot(query: str):
      # Verifica que nltk tenga los datos necesarios
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    # Dividimos la consulta en palabras clave, para entender mejor la intención del usuario
    #query_words = word_tokenize(query.lower())
     # Tokenizamos la consulta (verificando si nltk está correctamente instalado)
    try:
        tokenizer = RegexpTokenizer(r'\w+')  # Tokenizador que solo toma palabras
        query_words = tokenizer.tokenize(query.lower())
    except Exception as e:
        return JSONResponse(content={"error": f"Error al tokenizar la consulta: {str(e)}"}, status_code=500)
    
    # Buscamos sinónimos de las palabras clave para ampliar la búsqueda
    synonyms = {word for q in query_words for word in get_synonyms(q)} | set(query_words)
    
    # Filtramos la lista de pacientes buscando coincidencias en el servicio
    results = [m for m in pacientes_list if 'service' in m and isinstance(m['service'], str) and any(s in m['service'].lower() for s in synonyms)]
    
    # Si encontramos pacientes, enviamos la lista; si no, mostramos un mensaje de que no se encontraron coincidencias
    
    return JSONResponse (content={
        "respuesta": "Aquí tienes algunos pacientes relacionadosados." if results else "No encontré pacientes en ese servicio.",
        "pacientes": results
    })
    
# Ruta para buscar pacientes por servicio específico

@app.get ('/pacientes/by_service/', tags=['pacientes'])
def get_pacientes_by_service(service: str):
    # Filtramos la lista de pacientes según els ervicio ingresado
    return [m for m in pacientes_list if service.lower() in m['service'].lower()]