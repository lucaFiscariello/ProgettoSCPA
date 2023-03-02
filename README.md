# Obiettivo
Obiettivo del progetto è l'implementazione di un prodotto tra matrice sparsa multivettore sia in CUDA  che in OpenMP cercando di ottimizzare per quanto possibile le prestazioni computazionali. Le matrici sparse sono memorizzate nei formati:
- Ellpack
- CSR

# Modello
È stato adottato un approccio Object-Oriented applicato al C per poter modellare le astrazioni chiavi e i componenti del sistema. L'approccio Object-Oriented ha permesso l'implementazione dei seguenti pattern:
- mediator
- prototype
- state
- builder

Di seguito è riportato il modello implementato.

![state_builder_prototype](https://user-images.githubusercontent.com/80633764/222441134-ff269c7f-2f73-48f6-8916-dfd9f68d4f5c.png)

# Prestazioni
Per valutare la bontà delle prestazioni è stato preso in considerazione il modello Roofline che permette di individuare agevolmente gli upper bound prestazionali. Di seguito sono riportate le prestazioni dell'implementazione CUDA per i formati Ellpack e CSR.


![rooflineEllpack (1)](https://user-images.githubusercontent.com/80633764/222441864-17b3b855-ade9-4c8c-8a58-11104b18d82c.png)
![rooflineCSR (1)](https://user-images.githubusercontent.com/80633764/222441883-a718a157-47a7-46c9-959c-bd049d186396.png)

Infine vengono mostrati anche le prestazioni delle implementazioni OpenMP al variare del numero di thread allocati per entrambi i formati implementati.

![CSR_threads (1)](https://user-images.githubusercontent.com/80633764/222442424-25576615-2b34-43cf-b687-2cbc8007a333.png)
![Ellpack_threads (1)](https://user-images.githubusercontent.com/80633764/222442572-a68c0398-a769-4c5c-aade-eb2c903eac6b.png)
