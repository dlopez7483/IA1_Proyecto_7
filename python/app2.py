from tkinter import Tk, Label, Text, Scrollbar, Entry, Button
from tkinter import DISABLED, END, NORMAL
from chat import responder
from datetime import datetime

COLOR_FONDO_GRIS = "azure3"
COLOR_FONDO = "#FFFFFF"
COLOR_TEXTO = "#000000"
COLOR_FONDO_CHATBOT = "azure2"

FUENTE = ("Helvetica", 18)
FUENTE_NEGRITA = ("Helvetica", 14, "bold")

class AplicacionChat:
    def __init__(self):
        self.ventana = Tk()
        self._configurarVentanaPrincipal()

    def ejecutar(self):
        self.ventana.mainloop()

    def _configurarVentanaPrincipal(self):
        self.ventana.title("Proyecto Fase 3 - Grupo 7")
        self.ventana.resizable(width=True, height=True)

        anchoPantalla = self.ventana.winfo_screenwidth()
        altoPantalla = self.ventana.winfo_screenheight()

        x = (anchoPantalla // 2) - (1000 // 2)
        y = (altoPantalla // 2) - (800 // 2)

        self.ventana.geometry(f"{1000}x{800}+{x}+{y}")
        self.ventana.configure(background=COLOR_FONDO)

        lineaSeparadora = Label(self.ventana, bg=COLOR_FONDO_GRIS)
        lineaSeparadora.place(relwidth=1, rely=0.07, relheight=0.012)

        # Ventana de chat
        self.widgetTexto = Text(
            self.ventana,
            bg=COLOR_FONDO,
            fg=COLOR_TEXTO,
            font=FUENTE,
            padx=5,
            pady=5
        )
        self.widgetTexto.place(relheight=0.75, relwidth=1, rely=0.08)
        self.widgetTexto.configure(cursor="arrow", state=DISABLED)

        barraDesplazamiento = Scrollbar(self.widgetTexto)
        barraDesplazamiento.place(relheight=1, relx=0.974)
        barraDesplazamiento.configure(command=self.widgetTexto.yview)

        etiquetaInferior = Label(self.ventana, bg=COLOR_FONDO_GRIS, height=80)
        etiquetaInferior.place(relwidth=1, rely=0.825)

        self.entradaMensaje = Entry(etiquetaInferior, bg="#F0F0F0", fg=COLOR_TEXTO, font=FUENTE)
        self.entradaMensaje.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.entradaMensaje.focus()
        self.entradaMensaje.bind("<Return>", self._alPresionarEnter)

        botonEnviar = Button(
            etiquetaInferior,
            text="Enviar",
            font=FUENTE_NEGRITA,
            bg=COLOR_FONDO_GRIS,
            command=lambda: self._alPresionarEnter(None)
        )
        botonEnviar.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _alPresionarEnter(self, evento):
        mensaje = self.entradaMensaje.get()
        if mensaje.strip():
            self._insertarMensaje(mensaje, "Usuario")

    def _insertarMensaje(self, mensaje, remitente):
        self.entradaMensaje.delete(0, END)

        ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metaUsuario = f"**{remitente} [{ahora}]:**\n"
        self.widgetTexto.configure(state=NORMAL)
        self.widgetTexto.insert(END, metaUsuario, ("etiquetaMetaUsuario",))
        self.widgetTexto.insert(END, f"{mensaje}\n\n", ("etiquetaMensajeUsuario",))
        self.widgetTexto.tag_config("etiquetaMetaUsuario", justify="right", foreground=COLOR_TEXTO, font=FUENTE_NEGRITA, rmargin=40)
        self.widgetTexto.tag_config("etiquetaMensajeUsuario", justify="right", foreground=COLOR_TEXTO, rmargin=40)
        self.widgetTexto.configure(state=DISABLED)

        respuesta = responder(mensaje)
        metaChatbot = f"**Chatbot [{ahora}]:**\n"
        self.widgetTexto.configure(state=NORMAL)
        self.widgetTexto.insert(END, metaChatbot, ("etiquetaMetaChatbot",))
        self.widgetTexto.insert(END, f"{respuesta}\n\n", ("etiquetaMensajeChatbot",))
        self.widgetTexto.tag_config("etiquetaMetaChatbot", justify="left", background=COLOR_FONDO_CHATBOT, foreground=COLOR_TEXTO, font=FUENTE_NEGRITA, lmargin1=10, lmargin2=10, rmargin=40)
        self.widgetTexto.tag_config("etiquetaMensajeChatbot", justify="left", background=COLOR_FONDO_CHATBOT, foreground=COLOR_TEXTO, lmargin1=10, lmargin2=10, rmargin=40)
        self.widgetTexto.configure(state=DISABLED)

        self.widgetTexto.see(END)

if __name__ == "__main__":
    aplicacion = AplicacionChat()
    aplicacion.ejecutar()
