# ======================================================
# MÓDULOS Y CONFIGURACIÓN INICIAL
# ======================================================

# Bibliotecas estándar
import os
from typing import List, Dict, Union

# Bibliotecas de terceros
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
import unidecode

# Configuración de NLTK
nltk.data.path.append('C:\\Users\\latat\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\nltk')
nltk_path = os.path.join(os.getenv("APPDATA"), "nltk_data")
nltk.data.path.append(nltk_path)

# Descarga de recursos necesarios para NLP
nltk.download('punkt')  # Tokenizador de texto
nltk.download('wordnet')  # Base de datos léxica para sinónimos

# ======================================================
# CARGA Y PREPARACIÓN DE DATOS
# ======================================================

def load_pacientes() -> List[Dict]:
    """
    Carga y prepara los datos de pacientes desde un archivo CSV.
    
    Returns:
        List[Dict]: Lista de diccionarios con información de pacientes
    """
    try:
        # Cargar datos y seleccionar columnas relevantes
        df = pd.read_csv(
            "Dataset/Dataset_Pacientes_LOS.csv",
            delimiter=";",
            quotechar='"',
            on_bad_lines="skip"
        )[['id', 'fecha_entrada', 'fecha_alta', 'Genero', 'Edad', 'Enfermedad', 'Servicio', 'Estancia']]
        
        # Renombrar columnas para consistencia
        df.columns = ['id', 'fecha_entrada', 'fecha_alta', 'Genero', 'Edad (años)', 'Enfermedad', 'Servicio', 'Estancia (días)']
        
        return df.fillna('').to_dict(orient='records')
    
    except Exception as e:
        raise RuntimeError(f"Error cargando datos: {str(e)}")

# Cargar datos al iniciar la aplicación
pacientes_list: List[Dict] = load_pacientes()

# ======================================================
# FUNCIONALIDADES DE PROCESAMIENTO DE TEXTO
# ======================================================

def get_synonyms(word: str) -> set:
    """
    Genera sinónimos para una palabra usando WordNet.
    
    Args:
        word (str): Palabra para buscar sinónimos
        
    Returns:
        set: Conjunto de sinónimos en minúsculas
    """
    return {lemma.name().lower() for syn in wordnet.synsets(word) for lemma in syn.lemmas()}

# ======================================================
# CONFIGURACIÓN DE FASTAPI
# ======================================================

app = FastAPI(
    title="Sistema de Gestión Hospitalaria",
    description="API para gestión y consulta de pacientes hospitalizados",
    version="1.0.0"
)

# Configurar archivos estáticos (imágenes, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ======================================================
# ENDPOINTS PRINCIPALES
# ======================================================

@app.get('/', tags=['Home'])
def home():
    """Endpoint raíz que sirve la interfaz web principal"""
    html_content = """
    <html>
    <head>
        <title>API de Gestión Hospitalaria</title>
        <style>
            /* ============ ESTILOS BASE ============ */
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #D2D2D2;
            }

            h1 {
                text-align: center;
                color: #0033A0;
                font-size: 36px;
                margin-bottom: 30px;
            }

            /* ============ ESTRUCTURA PRINCIPAL ============ */
            .main-container {
                display: flex;
                gap: 30px;
                align-items: flex-start;
            }

            /* Panel izquierdo (Búsquedas y Resultados) */
            .left-panel {
                flex: 1;
                min-width: 500px;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }

            /* Panel derecho (Información e Imagen) */
            .right-panel {
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }

            /* ============ COMPONENTES DE BÚSQUEDA ============ */
            .search-box {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }

            .search-group {
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
            }

            .search-group input {
                flex: 1;
                padding: 12px;
                font-size: 16px;
                border-radius: 5px;
                border: 2px solid #0033A0;
            }

            .search-group input:focus {
                outline: 2px solid #0033A0;
            }

            .search-group button {
                padding: 12px 20px;
                background-color: #0033A0;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }

            /* ============ SECCIÓN DE RESULTADOS ============ */
            .search-results-container {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
                min-height: 200px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }

            /* ============ COMPONENTES VISUALES ============ */
            .info-section {
                background-color: #92A4DD;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }

            .hospital-image {
                width: 100%;
                max-width: 1200px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-top: 20px;
            }

            /* ============ CHATBOT ============ */
            #chatModal {
                display: none;
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 800px;
                background: #002366;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.5);
                z-index: 1000;
            }

            .chat-container {
                height: 300px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                margin-bottom: 10px;
            }

            .chat-message {
                margin: 5px 0;
                padding: 8px;
                border-radius: 5px;
            }

            .user-message {
                background: #B7C4EF;
                text-align: right;
                color: black;
            }

            .bot-message {
                background: #E6CFF7;
                text-align: left;
                color: black;
            }

            /* ============ BOTÓN DE ACTUALIZACIÓN ============ */
            .refresh-button {
                background-color: #0033A0;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background 0.3s;
            }

            .refresh-button:hover {
                background-color: green;
            }

            /* ============ TABLA DE RESULTADOS ============ */
            .results-table {
                width: 100%;
                border-collapse: collapse;
                background: #D2DCF;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                border-radius: 8px;
            }

            .results-table th {
                background-color:rgb(124, 79, 180);
                color: white;
                padding: 4px;
                text-align: center;
                position: sticky;
                top: 0;
                font-size: 15px;
            }

            .results-table td {
                padding: 4px;
                border-bottom: 1px solid #eee;
                color: #333;
                text-align: center;
            }

            .results-table tr:hover {
                background-color:rgb(171, 153, 180);
            }

            .no-results {
                padding: 20px;
                background: #ffeef0;
                color: #dc3545;
                border-radius: 8px;
                text-align: center;
            }
        </style>

        <!-- ============ SCRIPTS ============ -->
        <script>
            // Función principal para manejar búsquedas
            async function handleSearch(type) {
                const inputMap = {
                    'id': 'search-id',
                    'enfermedad': 'search-enfermedad',
                    'promedio': 'search-promedio',
                    'servicio': 'search-servicio'
                };
                
                const input = document.getElementById(inputMap[type]);
                const value = input.value.trim();
                let url = '';
                
                // Construcción de URLs según tipo de búsqueda
                switch(type) {
                    case 'id':
                        url = `/pacientes/${value}`;
                        break;
                    case 'enfermedad':
                        url = `/pacientes/por_enfermedad/?enfermedad=${value}`;
                        break;
                    case 'servicio':
                        url = `/pacientes/por_servicio/?servicio=${value}`;
                        break;
                    case 'promedio':
                        url = `/pacientes/promedio_estancia_por_enfermedad/?enfermedad=${value}`;
                        break;
                }

                if(value) {
                    try {
                        const response = await fetch(url);
                        const data = await response.json();
                        displayResults(data);
                    } catch(error) {
                        displayResults({ error: 'No se encontraron resultados' });
                    }
                }
            }

            // Función para mostrar resultados en formato tabla
            function displayResults(data) {
                const resultsDiv = document.getElementById('searchResults');
                let html = '';
                
                if (data.error || data.detalle) {
                    html = `<div class="no-results">${data.error || data.detalle}</div>`;
                } else if (Array.isArray(data) && data.length > 0) {
                    // Generar tabla para múltiples resultados
                    html = `<table class="results-table">
                                <thead>
                                    <tr>
                                        ${Object.keys(data[0]).map(key => `
                                            <th>${key.toUpperCase()}</th>
                                        `).join('')}
                                    </tr>
                                </thead>
                                <tbody>
                                    ${data.map(item => `
                                        <tr>
                                            ${Object.values(item).map(value => `
                                                <td>${typeof value === 'string' ? value.charAt(0).toUpperCase() + value.slice(1) : value}</td>
                                            `).join('')}
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>`;
                } else if (typeof data === 'object') {
                    // Generar tabla para resultado único
                    html = `<table class="results-table">
                                <tbody>
                                    ${Object.entries(data).map(([key, value]) => `
                                        <tr>
                                            <th>${key.replace(/_/g, ' ').toUpperCase()}</th>
                                            <td>${value}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>`;
                } else {
                    html = `<div class="no-results">No se encontraron coincidencias</div>`;
                }
                
                resultsDiv.innerHTML = html;
            }

            // Funcionalidad del Chatbot
            function showChatModal() {
                document.getElementById('chatModal').style.display = 'block';
            }

            function closeChatModal() {
                document.getElementById('chatModal').style.display = 'none';
            }

            async function handleChatQuery() {
                const input = document.getElementById('chatInput');
                const message = input.value.trim();
                const chatContainer = document.getElementById('chatContainer');
                
                if (message) {
                    chatContainer.innerHTML += `<div class="chat-message user-message">${message}</div>`;
                    
                    // Lógica de respuestas del chatbot
                    let response = '';
                    const greetings = ['hola', 'buenos dias', 'buenas tardes', 'buenas noches'];

                    if (greetings.some(greet => message.toLowerCase().includes(greet))) {
                        response = '¡Hola! Bienvenid@. 😊 ¡Qué alegría tenerte aquí! Soy tu asistente virtual y estoy listo para ayudarte en lo que necesites. 💡 Puedes preguntarme sobre <strong>nuestros servicios</strong>, <strong>cómo funciona la plataforma</strong> o cualquier otra duda que tengas. 🚀 ¡Empecemos! ¿En qué puedo ayudarte hoy?';

                    } else if (message.toLowerCase().includes('podrias darme el promedio de dias de estancia para la enfermedad diabetes?')) {
                        response = 'El promedio de días de estancia para esta enfermedad es de 5,57. Sin embargo, debes tener en cuenta que depende de la edad, el género y otros factores relacionados con la historia clínica del usuario.';

                    } else if (message.toLowerCase().includes('ahora, podrias decirme cuántas camas tenemos disponibles en este momento?')) {
                        response = 'Claro, en este momento disponemos de 10 camas disponibles en diferentes pabellones. ¿Quieres que indique el dato por pabellon?';

                    } else if (message.toLowerCase().includes('si')) {
                        response = 'Perfecto, ten cuento que el pabellon 2 tenemos 1 cama disponible; en el pabellon 7 tenemos 4 camas disponibles; en el pabellon 8 tenemos 3 camas disponibles; y en el pabellon 10 tenemos 2cama disponible.';

                    } else if (message.toLowerCase().includes('ok, podrias indicarme cuántos pacientes serán dados de alta el día de hoy?')) {
                        response = 'La cantidad de pacientes proyectados para dar de alta el día de hoy es: 5.';

                    } else {
                        response = 'Lo siento, no entiendo tu pregunta. ¿Podrías reformularla o especificar más detalles? 😊';
                    }

                    chatContainer.innerHTML += `<div class="chat-message bot-message">${response}</div>`;
                    input.value = '';
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }

            // Funciones auxiliares
            function handleEnter(event, type) {
                if (event.key === 'Enter') {
                    handleSearch(type);
                    event.preventDefault();
                }
            }

            function refreshPage() {
                location.reload();
            }
        </script>
    </head>

    <body>
        <!-- ============ HEADER ============ -->
        <h1>Bienvenido a la API de Pacientes Hospitalizados</h1>

        <!-- ============ CONTENIDO PRINCIPAL ============ -->
        <div class="main-container">
            <!-- Panel Izquierdo -->
            <div class="left-panel">
                <!-- Sección de Búsquedas -->
                <div class="search-box">
                    <div class="search-group">
                        <input type="text" id="search-id" placeholder="🔍 Buscar por ID del Paciente" 
                               onkeypress="handleEnter(event, 'id')">
                        <button onclick="handleSearch('id')" style="background: #3498DB; color: white; padding: 12px 20px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer;">Buscar</button>
                    </div>
                    <div class="search-group">
                        <input type="text" id="search-enfermedad" placeholder="🔍 Buscar por Enfermedad" 
                               onkeypress="handleEnter(event, 'enfermedad')">
                        <button onclick="handleSearch('enfermedad')" style="background: #3498DB; color: white; padding: 12px 20px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer;">Buscar</button>
                    </div>
                    <div class="search-group">
                        <input type="text" id="search-promedio" placeholder="🔍 Calcular Promedio días Estancia por Enfermedad" 
                               onkeypress="handleEnter(event, 'promedio')">
                        <button onclick="handleSearch('promedio')" style="background: #3498DB; color: white; padding: 12px 20px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer;">Buscar</button>
                    </div>
                    <div class="search-group">
                        <input type="text" id="search-servicio" placeholder="🔍 Buscar por Servicio (Hospitalización, Pediatría, Ginecología, Nanotología)" 
                               onkeypress="handleEnter(event, 'servicio')">
                        <button onclick="handleSearch('servicio')" style="background: #3498DB; color: white; padding: 12px 20px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer;">Buscar</button>
                    </div>
                    <button class="refresh-button" onclick="refreshPage()" style="background: #B7C4EF; color: black; padding: 12px 20px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer;">🔄 Actualizar</button>
                </div>

                <!-- Acceso al Chatbot -->
                <div class="chatbox">
                    💬 ¿Tienes más preguntas? ¡Habla con nuestro chatbot!
                    <button onclick="showChatModal()" style="background: #002366; color: white; padding: 12px 20px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer;">💬 Ir al Chatbot</button>
                </div>

                <!-- Sección de Resultados -->
                <div class="search-results-container">
                    <div id="searchResults"></div>
                </div>
            </div>

            <!-- Panel Derecho -->
            <div class="right-panel">
                <!-- Información Institucional -->
                <div class="info-section">
                    <p>Esta API permite gestionar de manera eficiente:</p>
                    <ul>
                        <li>✅ Hospitalización de pacientes</li>
                        <li>✅ Optimización de recursos médicos</li>
                        <li>✅ Planificación hospitalaria</li>
                        <li>✅ Toma de decisiones médicas</li>
                    </ul>
                    <p>También podrás obtener información importante sobre los pacientes según su patología y el servicio en el que se encuentran.</p>
                </div>

                <!-- Imagen Institucional -->
                <img src="/static/hospital_futurista.png" alt="Hospital futurista" class="hospital-image">
            </div>
        </div>

        <!-- ============ MODAL DEL CHATBOT ============ -->
        <div id="chatModal">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="color: white;">Chatbot Hospitalario</h3>
                <button onclick="closeChatModal()" class="refresh-button" style="background: #7C4FB4; color: white; padding: 12px 20px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer;">Cerrar</button>
            </div>

            <div class="chat-container" id="chatContainer" style="background: white; padding: 10px; border-radius: 5px; height: 300px; overflow-y: auto;"></div>
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <input type="text" id="chatInput" style="flex: 1; padding: 10px; font-size: 16px; border-radius: 5px; border: 1px solid #ccc; placeholder="Escribe tu pregunta...;"
                       onkeypress="if(event.key === 'Enter') handleChatQuery()">
                <button onclick="handleChatQuery()" style="background: #3498DB; color: white; padding: 12px 20px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer;">
                    Enviar
                </button>
            </div>
        </div>
    </body>
</html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get('/pacientes', tags=['pacientes'])
def get_pacientes():
    return pacientes_list or HTTPException(status_code=500, detail="No hay datos de pacientes hospitalizados")

@app.get('/pacientes/{id}', tags=['pacientes'])
def get_pacientes(id: str):
    return next((m for m in pacientes_list if m['id'] == id), {"detalle": "paciente no encontrado"})

@app.get('/pacientes/por_servicio/', tags=['pacientes'])
def get_pacientes_por_service(servicio: str):
    resultados = [m for m in pacientes_list if 'Servicio' in m and isinstance(m['Servicio'], str) and servicio.lower() in m['Servicio'].lower()]
    
    if not resultados:
        return {"mensaje": "No hay datos disponibles para el servicio"}
    
    return resultados

@app.get('/pacientes/por_estancia/', tags=['pacientes'])
def get_pacientes_por_Estancia(estancia: int):
    return [m for m in pacientes_list if isinstance(m['Estancia (días)'], (int, float)) and m['Estancia (días)'] == estancia]

@app.get('/pacientes/por_enfermedad/', tags=['pacientes'])
def get_pacientes_por_enfermedad(enfermedad: str):
    resultados = [m for m in pacientes_list if 'Enfermedad' in m and isinstance(m['Enfermedad'], str) and enfermedad.lower() in m['Enfermedad'].lower()]
    
    if not resultados:
        return {"mensaje": "No hay datos disponibles para la enfermedad"}
    
    return resultados

@app.get("/pacientes/promedio_estancia_por_enfermedad/", tags=["pacientes"])
def get_promedio_Estancia_por_enfermedad(enfermedad: str):
    estancia_list = []
    for paciente in pacientes_list:
        if paciente["Enfermedad"].lower() == enfermedad.lower():
            try:
                estancia = float(paciente["Estancia (días)"])
                if estancia > 0:
                    estancia_list.append(estancia)
            except ValueError:
                continue
    if not estancia_list:
        return {"message": f"No hay datos disponibles para la enfermedad '{enfermedad}'"}
    promedio_estancia = round(sum(estancia_list) / len(estancia_list), 2)
    return {"enfermedad": enfermedad, "promedio_Estancia_(días)": promedio_estancia}
