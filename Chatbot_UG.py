import spacy
import unicodedata
import re
import random
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning)

# Cargamos el modelo spaCy en español
nlp = spacy.load("es_core_news_sm")

# Lista global de carreras UG
carreras_ug = [
    # Ciencias e Ingeniería
    "alimentos", "arquitectura", "biología", "bioquímica farmacéutica",
    "ciencia de datos e inteligencia artificial",
    "geología", "ingeniería ambiental", "ingeniería civil",
    "ingeniería industrial", "ingeniería de la producción",
    "ingeniería química", "sistemas de información", "software",
    "tecnologías de la información", "telemática",
    # Agricultura
    "agronomía", "agropecuaria", "medicina veterinaria",
    # Artes
    "diseño gráfico", "diseño de interiores",
    # Programas Básicos, Educación, Servicios, Ciencias Sociales y Humanidades
    "ciencias políticas", "comunicación", "derecho", "educación básica",
    "educación inicial", "entrenamiento deportivo",
    "gastronomía", "pedagogía de la informática",
    "pedagogía de los idiomas nacionales y extranjeros",
    "pedagogía de la lengua y literatura",
    "pedagogía de la actividad física y deporte",
    "pedagogía de la química y biología",
    "pedagogía de la historia y ciencias sociales",
    "pedagogía de las artes y las humanidades",
    "pedagogía de las matemáticas y la física", "psicología",
    "publicidad", "sociología",
    # Educación Comercial, Economía y Afines
    "administración de empresas", "comercio exterior",
    "contabilidad y auditoría", "economía", "economía internacional",
    "finanzas", "gestión de la información gerencial", "mercadotecnia",
    "negocios internacionales", "turismo",
    # Salud
    "enfermería", "fonoaudiología", "medicina", "nutrición y dietética",
    "obstetricia", "odontología",
    "terapia ocupacional", "terapia respiratoria"
]


# Función de preprocesamiento
def preprocesar(texto):
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc if
              not token.is_stop and token.is_alpha]
    return " ".join(tokens)


# Dataset completo con utterances y etiquetas
datos = [
    # preguntar_requisitos_admision
    ("¿Cuáles son los requisitos para inscribirme en la UG?",
     "preguntar_requisitos_admision"),
    ("Qué documentos necesito para la admisión?",
     "preguntar_requisitos_admision"),
    ("Requisitos de ingreso a la Universidad de Guayaquil",
     "preguntar_requisitos_admision"),
    ("Qué piden para postularme?", "preguntar_requisitos_admision"),
    ("Necesito saber qué requisitos hay para la admisión",
     "preguntar_requisitos_admision"),
    ("¿Dónde puedo ver información sobre el proceso de admisión?",
     "preguntar_requisitos_admision"),
    ("Dónde encuentro detalles del proceso de admisión?",
     "preguntar_requisitos_admision"),
    ("Información sobre el proceso de admisión UG",
     "preguntar_requisitos_admision"),
    ("Dónde está la guía del proceso de admisión?",
     "preguntar_requisitos_admision"),
    ("Cómo me informo sobre el proceso de admisión?",
     "preguntar_requisitos_admision"),
    ("Me das mas info sobre la admision?", "preguntar_requisitos_admision"),

    # consultar_fechas_procesos (fusionado)
    ("¿Cuándo son las fechas de inscripción para admisión?",
     "consultar_fechas_procesos"),
    ("Fechas para registrarme en la UG", "consultar_fechas_procesos"),
    ("Dime cuándo abren las inscripciones de admisión",
     "consultar_fechas_procesos"),
    ("Necesito saber las fechas de postulación", "consultar_fechas_procesos"),
    ("Cuándo inicia el proceso de admisión", "consultar_fechas_procesos"),
    (
    "¿Cuándo empiezan las clases de nivelación?", "consultar_fechas_procesos"),
    ("Fechas de matrícula para nivelación", "consultar_fechas_procesos"),
    ("Dime cuándo inicia nivelación", "consultar_fechas_procesos"),
    ("Cuándo matriculo nivelación UG", "consultar_fechas_procesos"),
    ("Quiero saber las fechas de nivelación", "consultar_fechas_procesos"),

    # explicar_creacion_cuenta
    ("Cómo creo mi cuenta para admisión UG", "explicar_creacion_cuenta"),
    ("No sé crear la cuenta en la plataforma", "explicar_creacion_cuenta"),
    ("Ayúdame a crear cuenta para registrarme", "explicar_creacion_cuenta"),
    ("Crear usuario en plataforma de admisión", "explicar_creacion_cuenta"),
    ("Qué pasos sigo para crear la cuenta UG", "explicar_creacion_cuenta"),

    # consultar_cupos_carrera
    ("Hay cupos para medicina?", "consultar_cupos_carrera"),
    ("Cuántos cupos hay en derecho?", "consultar_cupos_carrera"),
    ("Cupos disponibles en ingeniería civil UG", "consultar_cupos_carrera"),
    ("Dime si hay cupo en enfermería", "consultar_cupos_carrera"),
    ("Quiero saber los cupos de odontología", "consultar_cupos_carrera"),

    # consultar_notas_aprobacion
    ("Qué nota necesito para aprobar nivelación?",
     "consultar_notas_aprobacion"),
    ("Cuál es la nota mínima para pasar nivelación UG",
     "consultar_notas_aprobacion"),
    (
    "Nota de aprobación en curso de nivelación", "consultar_notas_aprobacion"),
    ("Qué calificación aprueba nivelación", "consultar_notas_aprobacion"),
    ("Nota mínima para aprobar nivelación", "consultar_notas_aprobacion"),

    # consultar_asistencias_nivelacion
    ("Cuántas asistencias necesito en nivelación?",
     "consultar_asistencias_nivelacion"),
    ("Porcentaje mínimo de asistencia en nivelación",
     "consultar_asistencias_nivelacion"),
    ("Qué pasa si falto mucho en nivelación",
     "consultar_asistencias_nivelacion"),
    ("Cuántas faltas puedo tener en nivelación UG",
     "consultar_asistencias_nivelacion"),
    ("Asistencia requerida en nivelación", "consultar_asistencias_nivelacion"),
    ("Con cuantas asistencias se aprueba nivelación",
     "consultar_asistencias_nivelacion"),

    # consultar_resultados_examen
    ("Cuándo salen los resultados del examen de admisión?",
     "consultar_resultados_examen"),
    ("Resultados de examen UG", "consultar_resultados_examen"),
    ("Dime cuándo publican los resultados de admisión",
     "consultar_resultados_examen"),
    ("Fecha de resultados de examen", "consultar_resultados_examen"),
    (
    "Ya salieron los resultados de la prueba?", "consultar_resultados_examen"),

    # consultar_requisitos_matricula
    ("Qué requisitos hay para matricularme después de nivelación?",
     "consultar_requisitos_matricula"),
    ("Documentos para matriculación UG", "consultar_requisitos_matricula"),
    ("Requisitos de matrícula tras nivelación",
     "consultar_requisitos_matricula"),
    ("Qué piden para matricularme en la universidad",
     "consultar_requisitos_matricula"),
    ("Cómo hago la matrícula después de aprobar nivelación",
     "consultar_requisitos_matricula"),

    # saludar
    ("Hola", "saludar"),
    ("Buenas tardes", "saludar"),
    ("Buenos días", "saludar"),
    ("Qué tal", "saludar"),
    ("Hola chatbot", "saludar"),
    ("Hi", "saludar"),

    # consultar_becas
    ("¿Hay becas disponibles en la UG?", "consultar_becas"),
    ("Cómo puedo aplicar a una beca?", "consultar_becas"),
    ("Requisitos para becas UG", "consultar_becas"),
    ("Dame información sobre ayudas económicas", "consultar_becas"),
    ("Existen becas para estudiantes de nivelación?", "consultar_becas"),

    # consultar_contacto_soporte
    ("Cómo contacto a soporte de admisión?", "consultar_contacto_soporte"),
    ("Necesito ayuda, hay algún correo de soporte?",
     "consultar_contacto_soporte"),
    ("Teléfono de contacto UG", "consultar_contacto_soporte"),
    ("Dónde puedo pedir ayuda técnica?", "consultar_contacto_soporte"),
    ("Soporte para problemas en la plataforma", "consultar_contacto_soporte"),

    # consultar_proceso_apelacion
    ("Cómo apelo si no fui admitido?", "consultar_proceso_apelacion"),
    ("Proceso de apelación UG", "consultar_proceso_apelacion"),
    ("Qué hago si quiero reclamar mi resultado?",
     "consultar_proceso_apelacion"),
    ("Dónde presento una apelación?", "consultar_proceso_apelacion"),
    ("Apelación por resultados de admisión", "consultar_proceso_apelacion"),

    # consultar_examen_ingreso
    ("En qué consiste el examen de ingreso?", "consultar_examen_ingreso"),
    ("Temario del examen de admisión UG", "consultar_examen_ingreso"),
    ("Cuántas preguntas tiene el examen?", "consultar_examen_ingreso"),
    ("Duración del examen de ingreso", "consultar_examen_ingreso"),
    ("Qué materias entran en el examen?", "consultar_examen_ingreso"),

    # consultar_carreras (fusionado)
    ("Qué carreras ofrece la UG?", "consultar_carreras"),
    ("Lista de carreras disponibles", "consultar_carreras"),
    (
    "Carreras ofertadas en la Universidad de Guayaquil", "consultar_carreras"),
    ("Dime las carreras que puedo estudiar", "consultar_carreras"),
    ("Opciones de carreras en la UG", "consultar_carreras"),
    ("Cuáles son las carreras de ingeniería?", "consultar_carreras"),
    ("Dime las carreras de salud", "consultar_carreras"),
    ("Qué carreras hay en ciencias sociales", "consultar_carreras"),
    ("Carreras de educación", "consultar_carreras"),
    ("Opciones de ingeniería en la UG", "consultar_carreras"),
    ("Carreras de agricultura", "consultar_carreras"),
    ("Carreras de artes", "consultar_carreras"),
    # agradecer
    ("Gracias", "agradecer"),
    ("Muchas gracias", "agradecer"),
    ("Te agradezco", "agradecer"),
    ("thx", "agradecer"),
    ("thanks", "agradecer"),
    ("grax", "agradecer"),
    ("gracias bot", "agradecer"),
    ("mil gracias", "agradecer"),
    ("se agradece", "agradecer"),
    ("me das info sobre las carreras?", "consultar_carreras"),
    ("me das información sobre las carreras?", "consultar_carreras"),
    ("quiero información de las carreras", "consultar_carreras"),
    ("información de las carreras", "consultar_carreras"),
]

# Preprocesar textos
texts = [preprocesar(x[0]) for x in datos]
labels = [x[1] for x in datos]

# Vectorizar con TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Entrenar clasificador simple KNN
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X, labels)

# Diccionario de respuestas
respuestas = {
    "preguntar_requisitos_admision": "Para el proceso de admisión debes tener tu cédula, certificado de votación actualizado, título de bachiller y estar registrado en la plataforma de admisión.",
    "consultar_fechas_procesos": "Las fechas de inscripción y nivelación se publican en la página oficial de admisiones UG: https://admision.ug.edu.ec/admision/ y https://admision.ug.edu.ec/nivelacion/.",
    "explicar_creacion_cuenta": "Para crear tu cuenta ingresa a https://admision.ug.edu.ec/admision/, selecciona 'Registrarse' y completa los datos solicitados.",
    "consultar_cupos_carrera": "Los cupos por carrera se publican después del proceso de asignación. Consulta la página oficial o tu cuenta de admisión.",
    "consultar_notas_aprobacion": "Debes obtener al menos 7 sobre 10 para aprobar nivelación en la UG.",
    "consultar_asistencias_nivelacion": "Debes asistir al menos al 80% de clases para aprobar nivelación en la UG.",
    "consultar_resultados_examen": "Los resultados se publican en tu cuenta de admisión en la fecha oficial establecida. Revisa la página oficial.",
    "consultar_requisitos_matricula": "Para matricularte tras nivelación debes presentar cédula, certificado de votación, título de bachiller y aprobar asignaturas.",
    "saludar": "¡Hola! Soy el asistente virtual de admisiones UG. ¿En qué puedo ayudarte hoy?",
    "consultar_becas": "La UG ofrece becas y ayudas económicas según el rendimiento académico y situación socioeconómica. Consulta https://www.ug.edu.ec/becas/ para más información.",
    "consultar_contacto_soporte": "Puedes contactar a soporte de admisión al correo soporte.admision@ug.edu.ec o atencionalusuario@ug.edu.ec o al teléfono (04) 228-4505.",
    "consultar_proceso_apelacion": "El proceso de apelación se realiza en línea tras la publicación de resultados. Ingresa a tu cuenta y sigue las instrucciones en la sección de apelaciones.",
    "consultar_examen_ingreso": "El examen de ingreso evalúa conocimientos en matemáticas, lengua, ciencias y razonamiento. Consulta el temario oficial en la web de admisión.",
    "consultar_carreras": "La UG ofrece carreras en áreas de salud, ingeniería, ciencias sociales, educación, economía y más. Consulta la lista completa en https://admision.ug.edu.ec/oferta-ug/.",
    "fallback": "Lo siento, no entendí tu consulta. ¿Podrías reformularla?",
    "agradecer": [
        "Es un placer ayudarte, si tienes más preguntas no dudes en consultar.",
        "¡Con gusto! Si tienes otra consulta, aquí estaré.",
        "Para servirte, ¿hay algo más en lo que te pueda ayudar?",
        "¡De nada! Si necesitas más información, solo pregunta.",
        "Estoy para ayudarte, cualquier otra duda dime."
    ]
}

# Diccionario de carreras por categoría
carreras_por_categoria = {
    "ingeniería": [
        "Alimentos", "Arquitectura", "Biología", "Bioquímica Farmacéutica",
        "Ciencia de Datos e Inteligencia Artificial",
        "Geología", "Ingeniería Ambiental", "Ingeniería Civil",
        "Ingeniería Industrial", "Ingeniería de la Producción",
        "Ingeniería Química", "Sistemas de Información", "Software",
        "Tecnologías de la Información", "Telemática"
    ],
    "agricultura": [
        "Agronomía", "Agropecuaria", "Medicina Veterinaria"
    ],
    "artes": [
        "Diseño Gráfico", "Diseño de Interiores"
    ],
    "salud": [
        "Enfermería", "Fonoaudiología", "Medicina", "Nutrición y Dietética",
        "Obstetricia", "Odontología",
        "Terapia Ocupacional", "Terapia Respiratoria"
    ],
    "ciencias sociales": [
        "Ciencias Políticas", "Comunicación", "Derecho", "Psicología",
        "Publicidad", "Sociología"
    ],
    "educación": [
        "Educación Básica", "Educación Inicial", "Entrenamiento Deportivo",
        "Gastronomía", "Pedagogía de la Informática",
        "Pedagogía de los Idiomas Nacionales y Extranjeros",
        "Pedagogía de la Lengua y Literatura",
        "Pedagogía de la Actividad Física y Deporte",
        "Pedagogía de la Química y Biología",
        "Pedagogía de la Historia y Ciencias Sociales",
        "Pedagogía de las Artes y las Humanidades",
        "Pedagogía de las Matemáticas y la Física"
    ],
    "economía": [
        "Administración de Empresas", "Comercio Exterior",
        "Contabilidad y Auditoría", "Economía", "Economía Internacional",
        "Finanzas", "Gestión de la Información Gerencial", "Mercadotecnia",
        "Negocios Internacionales", "Turismo"
    ]
}


# Función para normalizar texto (quitar tildes y pasar a minúsculas)
def normalizar(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto.lower())
        if unicodedata.category(c) != 'Mn'
    )


# Funcion para predecir intent
def predecir_intent(texto_nuevo):
    texto_proc = preprocesar(texto_nuevo)
    X_nuevo = vectorizer.transform([texto_proc])
    pred = clf.predict(X_nuevo)
    return pred[0]


def predecir_intent_y_score(texto_nuevo):
    texto_proc = preprocesar(texto_nuevo)
    X_nuevo = vectorizer.transform([texto_proc])
    # Calcula la similitud (coseno) con todos los ejemplos
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(X_nuevo, X).flatten()
    max_sim = sims.max()
    pred = clf.predict(X_nuevo)
    return pred[0], max_sim


# Funcion para extraer entidades
def extraer_entidades(texto):
    doc = nlp(texto)
    entidades = []
    # Carreras
    for carrera in carreras_ug:
        if carrera in texto.lower():
            entidades.append((carrera, "CARRERA"))
    # Fechas (usando entidades de spaCy y regex)
    for ent in doc.ents:
        if ent.label_ in ["DATE"]:
            entidades.append((ent.text, "FECHA"))
    # Regex para fechas tipo dd/mm/aaaa o dd-mm-aaaa
    fechas_regex = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", texto)
    for fecha in fechas_regex:
        entidades.append((fecha, "FECHA"))
    return entidades


# Diccionario de variantes para cada categoría
variantes_categoria = {
    "ingeniería": ["ingenieria", "ingenierias", "ingenierías"],
    "salud": ["salud"],
    "artes": ["artes"],
    "ciencias sociales": ["ciencias sociales", "sociales"],
    "educación": ["educacion", "educaciones", "educación"],
    "agricultura": ["agricultura"],
    "economía": ["economia", "economias", "economía"]
}


# Función para responder
def responder(texto_usuario):
    intent, score = predecir_intent_y_score(texto_usuario)
    entidades = extraer_entidades(texto_usuario)
    UMBRAL = 0.3  # Puedes ajustar este valor
    if score < UMBRAL:
        return respuestas["fallback"]
    if intent in respuestas:
        respuesta = respuestas[intent]
        # Ejemplo de respuesta dinámica usando entidades
        if intent == "consultar_cupos_carrera":
            carreras = [e[0] for e in entidades if e[1] == "CARRERA"]
            if carreras:
                respuesta = f"Los cupos para {', '.join(carreras)} se publican después del proceso de asignación. Consulta la página oficial o tu cuenta de admisión."
        elif intent == "consultar_carreras":
            texto_norm = normalizar(texto_usuario)
            for cat, variantes in variantes_categoria.items():
                for var in variantes:
                    if var in texto_norm:
                        lista = carreras_por_categoria[cat]
                        return f"Las carreras de {cat} en la UG son: {', '.join(lista)}."
            # Si no se especifica categoría, lista general
            return respuesta
        # Si es agradecimiento, elegir respuesta aleatoria
        elif intent == "agradecer":
            return random.choice(respuesta)
        return respuesta
    else:
        return respuestas["fallback"]


# Evaluación básica
test_texts = [
    # Ejemplos correctos
    "Qué documentos necesito para postular a la UG?",
    "Cuándo empieza la matrícula para nivelación?",
    "Hola chatbot",
    "Qué nota necesito en nivelación?",
    "Cómo hago la cuenta para admisión?",
    "Cuántos cupos hay en derecho?",
    "No entiendo cómo crear cuenta",
    "Cuándo publican resultados?",
    "Cuántas faltas puedo tener en nivelación?",
    "Qué piden para matricularme?",
    "Cómo puedo aplicar a una beca?",
    "Teléfono de contacto UG",
    "Dónde presento una apelación?",
    "Temario del examen de admisión UG",
    "Dime las carreras que puedo estudiar",
    # Errores tipográficos
    "Quiero sabr los cupos de odontologia",
    "Cuand inicia el proceso de admision?",
    "Que requsitos hay para la admision?",
    "Como creo mi cuanta para admision UG",
    "Cuantas asitencias necesito en nivelacion?",
    # Preguntas ambiguas
    "Dame información de la universidad",
    "Necesito ayuda",
    "Quiero saber fechas",
    "Hay opciones para estudiantes?",
    "Qué puedo estudiar?"
]
test_labels = [
    "preguntar_requisitos_admision",
    "consultar_fechas_procesos",
    "saludar",
    "consultar_notas_aprobacion",
    "explicar_creacion_cuenta",
    "consultar_cupos_carrera",
    "explicar_creacion_cuenta",
    "consultar_resultados_examen",
    "consultar_asistencias_nivelacion",
    "consultar_requisitos_matricula",
    "consultar_becas",
    "consultar_contacto_soporte",
    "consultar_proceso_apelacion",
    "consultar_examen_ingreso",
    "consultar_carreras",
    # Errores tipográficos (esperado: intent correcto)
    "consultar_cupos_carrera",
    "consultar_fechas_procesos",
    "preguntar_requisitos_admision",
    "explicar_creacion_cuenta",
    "consultar_asistencias_nivelacion",
    # Ambiguas (esperado: fallback)
    "fallback",
    "consultar_contacto_soporte",
    "consultar_fechas_procesos",
    "consultar_becas",
    "consultar_carreras"
]

predicciones = [predecir_intent(texto) for texto in test_texts]

# Calcular accuracy
print("Accuracy:", accuracy_score(test_labels, predicciones))

# Calcular precision, recall, f1 para intents (macro)
labels_unicos = list(set(test_labels))
print("Precision (macro):",
      precision_score(test_labels, predicciones, average='macro',
                      zero_division=0))
print("Recall (macro):",
      recall_score(test_labels, predicciones, average='macro',
                   zero_division=0))
print("F1 (macro):",
      f1_score(test_labels, predicciones, average='macro', zero_division=0))

# Evaluación de entidades en el set de test
entidades_reales = []
entidades_predichas = []
for texto in test_texts:
    # Simulación: solo buscamos carreras y fechas en la frase
    carreras = [c for c in [
        "medicina", "derecho", "ingeniería civil", "enfermería", "odontología",
        "psicología", "administración", "contabilidad", "educación",
        "computación"
    ] if c in texto.lower()]
    fechas = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", texto)
    entidades_reales.append(set(carreras + fechas))
    entidades_predichas.append(set([e[0] for e in extraer_entidades(texto) if
                                    e[1] in ["CARRERA", "FECHA"]]))


# Calcular recall y precision para entidades (carreras y fechas)
def safe_div(a, b):
    return a / b if b else 0


true_positives = sum(
    [len(r & p) for r, p in zip(entidades_reales, entidades_predichas)])
false_positives = sum(
    [len(p - r) for r, p in zip(entidades_reales, entidades_predichas)])
false_negatives = sum(
    [len(r - p) for r, p in zip(entidades_reales, entidades_predichas)])

precision_ent = safe_div(true_positives, true_positives + false_positives)
recall_ent = safe_div(true_positives, true_positives + false_negatives)
print(f"Precision entidades (carreras y fechas): {precision_ent:.2f}")
print(f"Recall entidades (carreras y fechas): {recall_ent:.2f}")

# Conversación en consola
if __name__ == "__main__":
    print("\nChatbot Admisiones UG - Escribe 'salir' para terminar")
    estado = None
    while True:
        entrada = input("Tú: ")
        if entrada.lower() in ["salir", "adiós", "adios", "bye", "chao"]:
            print("Bot: ¡Hasta luego!")
            break
        # Manejo de estado conversacional para carreras
        if estado == "preguntar_carreras":
            if entrada.strip().lower() in ["sí", "si", "ok", "dale", "quiero",
                                           "muestralas", "muéstralas", "ver",
                                           "dime"]:
                print(
                    f"Bot: Las 56 carreras de la UG son: {', '.join(carreras_ug)}.")
                estado = None
                continue
            else:
                print(
                    "Bot: De acuerdo, si necesitas la lista de carreras, solo dime 'sí'.")
                estado = None
                continue
        # Manejo de estado conversacional para requisitos admisión
        if estado == "mas_requisitos_admision":
            if entrada.strip().lower() in ["sí", "si", "ok", "dale", "quiero",
                                           "ver", "dime", "más", "mas"]:
                print(
                    "Bot: Puedes consultar más información aquí: https://blog.alau.org/todo-lo-que-necesitas-saber-para-ingresar-a-la-universidad-de-guayaquil/")
                estado = None
                continue
            else:
                print(
                    "Bot: De acuerdo, si necesitas más información, no dudes en preguntar.")
                estado = None
                continue
        # Respuesta normal
        respuesta = responder(entrada)
        # Si la respuesta es sobre carreras, activar estado SOLO si el usuario pregunta por la cantidad o por la lista general
        if respuesta == respuestas["consultar_carreras"]:
            texto_norm = normalizar(entrada)
            if (
                    "cuántas carreras" in texto_norm or "cuantas carreras" in texto_norm or
                    "cuales son las carreras de la ug" in texto_norm or "cuáles son las carreras de la ug" in texto_norm
                    or "cuales son las carreras?" in texto_norm or "cuantas carreras hay" in texto_norm):
                print(
                    f"Bot: La UG ofrece 56 carreras. ¿Quieres saber cuáles son?")
                estado = "preguntar_carreras"
            else:
                print("Bot:", respuesta)
        # Si la respuesta es sobre requisitos de admisión, preguntar si quiere saber más
        elif respuesta == respuestas["preguntar_requisitos_admision"]:
            print("Bot:", respuesta)
            print("Bot: ¿Quieres saber más sobre el proceso de admisión?")
            estado = "mas_requisitos_admision"
        else:
            print("Bot:", respuesta)
