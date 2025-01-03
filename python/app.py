import tkinter as tk
from tkinter import scrolledtext
from chat import responder
from datetime import datetime

def enviarMensaje(evento=None):
    mensaje = entradaMensaje.get()
    if mensaje.strip():
        agregarMensajeAlChat("Usuario", mensaje, "floral white", "right")
        entradaMensaje.delete(0, tk.END)
        respuesta = responder(mensaje)
        agregarMensajeAlChat("Chatbot", respuesta, "ivory2", "left")

def agregarMensajeAlChat(remitente, mensaje, colorFondo, alineacion):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ventanaChat.configure(state='normal')
    textoEncabezado = f"{remitente} [{ahora}]:\n"
    ventanaChat.insert(tk.END, textoEncabezado, f"encabezado_{alineacion}")
    indiceInicio = f"{ventanaChat.index('end')}-2l linestart"
    indiceFin = f"{ventanaChat.index('end')}-1l lineend"
    ventanaChat.tag_add(remitente, indiceInicio, indiceFin)
    ventanaChat.tag_config(f"encabezado_{alineacion}", justify=alineacion, background=colorFondo)
    ventanaChat.insert(tk.END, f"{mensaje}\n", f"mensaje_{alineacion}")
    ventanaChat.tag_config(f"mensaje_{alineacion}", justify=alineacion)
    ventanaChat.see(tk.END)
    ventanaChat.configure(state='disabled')


raiz = tk.Tk()
raiz.title("Proyecto Fase 3 - Grupo 7")

anchoPantalla = raiz.winfo_screenwidth()
altoPantalla = raiz.winfo_screenheight()
raiz.geometry(f"{anchoPantalla}x{altoPantalla}")

raiz.configure(bg="white")


FUENTE_TITULO = ("Helvetica", 18, "bold")
FUENTE_TEXTO = ("Arial", 14, "bold")
FUENTE_ENTRADA = ("Arial", 12)


contenedorChat = tk.Frame(raiz, bg="white", highlightbackground="gray", highlightthickness=2)
contenedorChat.pack(padx=200, pady=100, fill=tk.BOTH, expand=True)


ventanaChat = scrolledtext.ScrolledText(contenedorChat, wrap=tk.WORD, font=FUENTE_TEXTO, bg="white", fg="black", state='disabled')
ventanaChat.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)


marcoEntrada = tk.Frame(contenedorChat, bg="white")
marcoEntrada.pack(fill=tk.X, padx=200, pady=20)

entradaMensaje = tk.Entry(marcoEntrada, font=FUENTE_ENTRADA, bg="light steel blue", fg="black")
entradaMensaje.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

entradaMensaje.bind("<Return>", enviarMensaje)

botonEnviar = tk.Button(marcoEntrada, text=">", font=FUENTE_ENTRADA, bg="steel blue", fg="white", command=enviarMensaje)
botonEnviar.pack(side=tk.RIGHT, padx=5, pady=5)


raiz.mainloop()
