@ -0,0 +1,385 @@
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QColorDialog, QFrame, QFileDialog, QMessageBox
from PyQt5.QtGui import QPainter, QPen, QBrush, QPolygon
from PyQt5.QtCore import Qt, QPoint
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class MyCNN(nn.Module):
    """
    Définition du réseau de neuronnes
    """

    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3) # Convolution avec kernel 3x3 et 16 sorties
        self.pool = nn.MaxPool2d(2, 2)  # MaxPooling 2x2
        self.conv2 = nn.Conv2d(16, 32, 4) # Convolution avec kernel 4x4 et 32 sorties

        # Couches entièrement connectées
        # W'=((W-F+2P)/S) + 1 (Width, Filter size, Padding, Stride)
        self.fc1 = nn.Linear(32*6*6, 512)  # sorties x W' x W'
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 195) 
        
    def forward(self, x):

        # MaxPooling sur la conv1 avec activation ReLu
        x = self.pool(F.relu(self.conv1(x)))
        # MaxPooling sur la conv2 avec activation ReLu
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
class Prediction():

    def load_model():

        #Impotation du modèle
        MODELS_DIR = 'models'
        best_checkpoint = torch.load(f'{MODELS_DIR}/mycnn_model_all_flags.pt')
        best_cnn = MyCNN()
        best_cnn.load_state_dict(best_checkpoint['model_state_dict'])

        return best_cnn

    def cnn_predict(file_path):

        # Chargement de l'image
        image = Image.open(file_path).convert('RGB')

        # moyenne et ecart-type
        image_mean = [0.4968, 0.4211, 0.3947]
        image_std = [0.0780, 0.0667, 0.0608]
        input_transform = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=image_mean, std=image_std)])

        # Création d'un tenseur
        img_tensor = input_transform(image).unsqueeze(0)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Perdiction avec le cnn
        best_cnn = Prediction.load_model()
        logodd = best_cnn(img_tensor.to(device))
        prob = torch.nn.functional.softmax(logodd, dim=1)[0]
        _, indices = torch.sort(prob, descending=True)

        return prob, indices

class PaintArea(QFrame):
    """ Définition de la zone de dessin"""

    def __init__(self):
        super().__init__()
        # Définition de la zone de dessin
        self.setFixedSize(512, 512)
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(2)
        self.setStyleSheet("background-color: white;")
        
        # Liste pour stocker toutes les formes dessinées
        self.shapes = []
        self.current_shape = None
        self.drawing = False
        self.shape_type = 'pen'
        self.fill_shapes = True
        self.pen_color = Qt.black

    def set_shape_type(self, shape_type):
        """Définition du type de forme à dessiner"""
        self.shape_type = shape_type

    def set_pen_color(self, color):
        """Définition de la couleur du crayon"""
        self.pen_color = color

    def mousePressEvent(self, event):
        """ Action lorsque la souris est cliquée"""

        # Dessine si le le bouton gauche de la souris est pressé
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos() # position de la souris
            if self.shape_type == 'pen':
                self.current_shape = {
                    'type': self.shape_type, 
                    'points': [self.start_point], 
                    'color': self.pen_color
                }
            else:
                self.current_shape = {
                    'type': self.shape_type, 
                    'start': self.start_point, 
                    'end': self.start_point, 
                    'fill': self.fill_shapes, 
                    'color': self.pen_color
                }

    def mouseMoveEvent(self, event):
        """ Action lorsque la souris bouge"""

        # Commence le dessin
        if self.drawing:
            if self.shape_type == 'pen':
                self.current_shape['points'].append(event.pos())
            else:
                self.current_shape['end'] = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """ Action lorsque la souris est relachée"""

        # Termine le dessin si le bouton gauche de la souris est relâché
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False  
            if self.shape_type != 'pen':
                self.current_shape['end'] = event.pos()  # MAJ du point de fin
            self.shapes.append(self.current_shape)
            
            # Réinitialise la forme actuelle
            self.current_shape = None 
            self.update()

    def paintEvent(self, event):
        """ Permet de peindre suivant les formes sélectionnées """
        
        # Crée un objet QPainter pour dessiner
        painter = QPainter(self)

        # Ajoute les formes au dessin
        for shape in self.shapes:

            # Propriété du crayon
            painter.setPen(QPen(shape['color'], 5, Qt.SolidLine))
            
            # Si "l'outil" est un "rayon
            if shape['type'] == 'pen':
                for i in range(len(shape['points']) - 1):
                    # Dessine des lignes entre les points
                    painter.drawLine(shape['points'][i], shape['points'][i + 1])

             # Si la forme est un rectangle
            elif shape['type'] == 'rectangle':
                
                # Remplit le rectangle
                if shape['fill']:
                    painter.setBrush(QBrush(shape['color'], Qt.SolidPattern))
                else:
                    painter.setBrush(Qt.NoBrush)
                
                # Dessine le rectangle    
                painter.drawRect(
                    shape['start'].x(),
                    shape['start'].y(), 
                    shape['end'].x() - shape['start'].x(), 
                    shape['end'].y() - shape['start'].y()
                )

             # Si la forme est un cercle
            elif shape['type'] == 'circle':

                # Calcule du rayon
                radius = ((shape['end'].x() - shape['start'].x())**2 + 
                          (shape['end'].y() - shape['start'].y())**2)**0.5
                
                # Remplit le cercle
                if shape['fill']:
                    painter.setBrush(QBrush(shape['color'], Qt.SolidPattern))
                else:
                    painter.setBrush(Qt.NoBrush)

                # Dessine le cercle
                painter.drawEllipse(shape['start'], int(radius), int(radius))

            # Si la forme est un triangle
            elif shape['type'] == 'triangle':  

                # Remplit le triangle
                if shape['fill']:
                    painter.setBrush(QBrush(shape['color'], Qt.SolidPattern))
                else:
                    painter.setBrush(Qt.NoBrush)

                # Définition des points du triangle
                points = [
                    shape['start'],
                    QPoint(shape['end'].x(), shape['start'].y()),
                    QPoint((shape['start'].x() + shape['end'].x()) // 2, shape['end'].y())
                ]

                # Dessine le triangle
                painter.drawPolygon(QPolygon(points))

        # Affichage les formes en "temps réel"
        if self.current_shape:
            painter.setPen(QPen(self.current_shape['color'], 5, Qt.SolidLine))

            # Si "l'outil" est un "rayon
            if self.current_shape['type'] == 'pen':
                for i in range(len(self.current_shape['points']) - 1):
                    painter.drawLine(self.current_shape['points'][i], self.current_shape['points'][i + 1])

            # Si la forme est un rectangle
            elif self.current_shape['type'] == 'rectangle':  
                if self.current_shape['fill']:
                    painter.setBrush(QBrush(self.current_shape['color'], Qt.SolidPattern))  # Remplit le rectangle si nécessaire
                else:
                    painter.setBrush(Qt.NoBrush)
                painter.drawRect(
                    self.current_shape['start'].x(),
                    self.current_shape['start'].y(),
                    self.current_shape['end'].x() - self.current_shape['start'].x(),
                    self.current_shape['end'].y() - self.current_shape['start'].y()
                )
            # Si la forme est un cercle
            elif self.current_shape['type'] == 'circle':  
                radius = ((self.current_shape['end'].x() - self.current_shape['start'].x())**2 + 
                          (self.current_shape['end'].y() - self.current_shape['start'].y())**2)**0.5
                if self.current_shape['fill']:
                    painter.setBrush(QBrush(self.current_shape['color'], Qt.SolidPattern))  # Remplit le cercle si nécessaire
                else:
                    painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(self.current_shape['start'], int(radius), int(radius))
            # Si la forme est un triangle
            elif self.current_shape['type'] == 'triangle':
                if self.current_shape['fill']:
                    painter.setBrush(QBrush(self.current_shape['color'], Qt.SolidPattern))
                else:
                    painter.setBrush(Qt.NoBrush)
                points = [
                    self.current_shape['start'],
                    QPoint(self.current_shape['end'].x(), self.current_shape['start'].y()),
                    QPoint((self.current_shape['start'].x() + self.current_shape['end'].x()) // 2, self.current_shape['end'].y())
                ]
                painter.drawPolygon(QPolygon(points))

    def save_image(self):
        """ Sauvegarde et prédction du dessin """

        # Définit le chemin de sauvegarde
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG (*.png);;All Files (*)")
        
        if file_path:

            # Sauvegarde l'image
            image = self.grab()
            image.save(file_path, "PNG")
            
            # Récupère les iso des pays
            FLAG_LIST_DIR = 'flag_list.txt'
            with open(FLAG_LIST_DIR, 'r') as f:
                iso = [line[:-1] for line in f] 
            
            # prédiction
            prob, indices = Prediction.cnn_predict(file_path)
    
            summary = []

            # Print des résultats et sauvegarde des trois premiers
            for idx,i in enumerate(indices):
                print(iso[i], f': {prob[i].item():.2%}')
                if idx in range(4):
                    summary.append(f'{iso[i]}: {prob[i].item():.2%}')
            print(summary)

            # PopUp avec les trois meilleurs prédictions
            QMessageBox.information(self, "Image Saved", f'Prediction:\n{summary[0]}\n{summary[1]}\n{summary[2]}')

class MainWindow(QMainWindow):
    """ Définition de fenêtre globale"""

    def __init__(self):
        super().__init__()

        # Paramètres de la fenêtre
        self.setWindowTitle('Simple Paint App')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: lightgrey;")

        # Définition de la zone de dessin
        self.paint_area = PaintArea()
        
        # Définition des boutons
        self.pen_button = QPushButton('Pen')
        self.pen_button.clicked.connect(self.set_pen_mode)
        
        self.rect_button = QPushButton('Rectangle')
        self.rect_button.clicked.connect(self.set_rect_mode)
        
        self.circle_button = QPushButton('Circle')
        self.circle_button.clicked.connect(self.set_circle_mode)
        
        self.triangle_button = QPushButton('Triangle')
        self.triangle_button.clicked.connect(self.set_triangle_mode)
        
        self.color_button = QPushButton('Choose Color')
        self.color_button.clicked.connect(self.choose_color)
        
        self.save_button = QPushButton('Save Image')
        self.save_button.clicked.connect(self.save_image)
        
        # Ajout de la zone de dessin à la fenêtre
        layout = QVBoxLayout()
        layout.addWidget(self.paint_area)
        
        # Ajouts des boutons à la fenêtre
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.pen_button)
        button_layout.addWidget(self.rect_button)
        button_layout.addWidget(self.circle_button)
        button_layout.addWidget(self.triangle_button)
        button_layout.addWidget(self.color_button)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
        
        # Crée un widget conteneur avec le layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def set_pen_mode(self):
        """ Définition du crayon comme mode de dessin"""
        self.paint_area.set_shape_type('pen')

    def set_rect_mode(self):
        """ Définition du rectangle comme mode de dessin"""
        self.paint_area.set_shape_type('rectangle')

    def set_circle_mode(self):
        """ Définition du cercle comme mode de dessin"""
        self.paint_area.set_shape_type('circle')

    def set_triangle_mode(self):
        """ Définition du triangle comme mode de dessin"""
        self.paint_area.set_shape_type('triangle')

    def choose_color(self):
        """ Défintion de la couleur"""

        # Ouvre la fenêtre de couleur
        color = QColorDialog.getColor()
        if color.isValid():
            self.paint_area.set_pen_color(color)

    def save_image(self):
        """ Sauvegarde du dessin"""
        self.paint_area.save_image()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()