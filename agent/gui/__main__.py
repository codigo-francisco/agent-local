from agent.gui.app import main

# Solo el proceso principal: la ventana nativa se abre en un subproceso que reimporta este módulo.
if __name__ == "__main__":
    main()
