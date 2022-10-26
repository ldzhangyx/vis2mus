import os
import sys
from ui_prepare import *


class Window(QWidget):
    switch_window = pyqtSignal()

    def __init__(self, text):
        super().__init__()
        self.setWindowTitle(text)
        self.move(100, 50)
        self.setup_ui()

    def setup_ui(self):

        layout = QHBoxLayout()

        gl = QGridLayout()
        layout4 = QVBoxLayout()

        layout.addLayout(gl, 3)
        layout.addLayout(layout4, 1)

        self.setLayout(layout)
        self.qpm2 = 90

        def stylize():

            reset_all_player()

            mel_notes = muspy.to_note_representation(muspy.read_midi('./resources/content.midi'))
            acc_notes = longx_to_notes(self.modified_long_x2)

            if acc_notes.shape[0]  >= 4:
                num = int(acc_notes.shape[0] / 4)

                self.shift = - int(mel_notes.mean(0)[1] - sum(sorted(acc_notes[:, 1], reverse=True)[0:num]) / num)
            else:
                self.shift = 0

            d = {
                1: -1,
                2: -2,
                3: +2,
                4: +1,
                6: +1,
                8: -1,
                9: -2,
                10: +2,
                11: +1
            }

            if self.shift % 12 in d:
                self.shift += d[self.shift % 12]

            self.modified_long_c1 = shift_chord(self.modified_long_c1, self.shift)

            c11 = self.modified_long_c1[:, 0:8, :]
            c12 = self.modified_long_c1[:, 8:16, :]
            c13 = self.modified_long_c1[:, 16:24, :]
            c14 = self.modified_long_c1[:, 24:32, :]

            pr_mat21 = self.modified_long_pr_mat2[:, 0, :, :]
            pr_mat22 = self.modified_long_pr_mat2[:, 1, :, :]
            pr_mat23 = self.modified_long_pr_mat2[:, 2, :, :]
            pr_mat24 = self.modified_long_pr_mat2[:, 3, :, :]

            out_x1 = model.swap(None, pr_mat21, c11, None, False, True)
            out_x2 = model.swap(None, pr_mat22, c12, None, False, True)
            out_x3 = model.swap(None, pr_mat23, c13, None, False, True)
            out_x4 = model.swap(None, pr_mat24, c14, None, False, True)

            out_x = np.stack((out_x1, out_x2, out_x3, out_x4), 1)

            merge_x_to_midi(None, self.shift, out_x, './resources/syn', self.qpm2)
            syn_player.set_qpm(self.qpm2)
            merge_x_visualization(None, self.shift, out_x, './resources/syn_pr')

            syn_player.findChild(QLabel, "pr").setPixmap(QPixmap('./resources/syn_pr.png'))
            syn_player.show()

            btn2.setEnabled(False)

        def stylize_img():
            path = './resources/syn_img/' + content_cb2.currentText() + '+' + style_cb.currentText() + '.jpg'
            syn_img.setPixmap(QPixmap(path))

        def reset_all_player():
            content_player.reset()
            style_player.reset()
            syn_player.reset()
            pygame.mixer.music.stop()

        def next_content(item):

            reset_all_player()

            s1.slider.setValue(0)
            s2.slider.setValue(0)

            items = [item + str(i) for i in range(1, 4)]  # item1: original; item2&3: variation
            content_cb2.blockSignals(True)
            content_cb2.clear()
            content_cb2.blockSignals(False)
            content_cb2.addItems(items)

            btn2.show()
            btn2.setEnabled(True)
            btn3.setEnabled(False)

        def content_changed(item):

            reset_all_player()
            syn_player.hide()

            s1.slider.setValue(0)
            s2.slider.setValue(0)

            if style_cb.currentText():
                next_style(style_cb.currentText())
            style_player.Is_Null = False

            path = './resources/content_img/' + item + '.png'
            content_img.setPixmap(QPixmap(path))

            self.long_x1, self.long_c1, self.long_pr_mat1, self.long_mel1, self.qpm1 = torch.load(
                './resources/content_music/' + item + '.pt')
            numpy_to_midi(self.long_mel1.reshape(-1, 130), './resources/content.midi', self.qpm2)
            content_player.set_qpm(self.qpm2)
            mel_visualization('./resources/content.midi')

            content_player.findChild(QLabel, "pr").setPixmap(QPixmap('./resources/content_pr.png'))

            self.modified_long_pr_mat1 = self.long_pr_mat1
            self.modified_long_c1 = self.long_c1
            stylize_img()


        def next_style(item):

            reset_all_player()
            syn_player.hide()

            s1.slider.setValue(0)
            s2.slider.setValue(0)

            path = './resources/style_img/' + item + '.jpg'
            style_img.setPixmap(QPixmap(path))

            self.long_x2, self.long_c2, self.long_pr_mat2, self.long_mel2, self.qpm2 = torch.load(
                './resources/style_music/' + item + '.pt')
            longx_to_midi(self.long_x2, './resources/style', self.qpm2)
            style_player.set_qpm(self.qpm2)
            longx_visualization(self.long_x2, './resources/style_pr')
            style_player.findChild(QLabel, "pr").setPixmap(QPixmap('./resources/style_pr.png'))
            style_player.Is_Null = False
            style_player.show()
            stylize_img()

            numpy_to_midi(self.long_mel1.reshape(-1, 130), './resources/content.midi', self.qpm2)
            content_player.set_qpm(self.qpm2)


            btn2.show()
            btn2.setEnabled(True)
            btn3.setEnabled(False)

            self.modified_long_x2 = self.long_x2
            self.modified_long_pr_mat2 = self.long_pr_mat2

        def img_edit():

            reset_all_player()
            btn2.hide()
            syn_player.hide()
            style_player.hide()

            style_img_path = './resources/style_img/' + style_cb.currentText() + '.jpg'
            syn_img_path = './resources/syn_img/' + content_cb2.currentText() + '+' + style_cb.currentText() + '.jpg'

            val1 = s1.val() / 100 + 1

            if s2.val() < 0:
                val2 = s2.val() / 100 + 1
            else:
                val2 = s2.val() / 10 + 1

            img1 = Image.open(style_img_path)
            enh_bri = ImageEnhance.Brightness(img1)
            image_brightened = enh_bri.enhance(val1)
            enh_con = ImageEnhance.Contrast(image_brightened)
            image_contrasted = enh_con.enhance(val2)

            image_contrasted.save('./resources/StyleImg_Enhanced.png')
            style_img.setPixmap(QPixmap('./resources/StyleImg_Enhanced.png'))

            img2 = Image.open(syn_img_path)
            enh_bri = ImageEnhance.Brightness(img2)
            image_brightened = enh_bri.enhance(val1)
            enh_con = ImageEnhance.Contrast(image_brightened)
            image_contrasted = enh_con.enhance(val2)

            image_contrasted.save('./resources/SynImg_Enhanced.png')
            syn_img.setPixmap(QPixmap('./resources/SynImg_Enhanced.png'))

            btn3.setEnabled(True)

        def fine_tune():

            reset_all_player()

            if s1.val() == 0 and s2.val() == 0:
                out_x = self.long_x2
            else:
                out_x = long_modify(self.long_x2, self.long_c2, self.long_pr_mat2, s1.val(), s2.val(), 0)

            self.modified_long_x2 = out_x
            self.modified_long_pr_mat2 = lx_to_lpr(out_x)
            acc_notes = longx_to_notes(self.modified_long_x2)
            if len(acc_notes) == 0:
                style_player.Is_Null = True
            else:
                style_player.Is_Null = False

            longx_to_midi(self.modified_long_x2, './resources/style', self.qpm2)
            style_player.set_qpm(self.qpm2)
            longx_visualization(self.modified_long_x2, './resources/style_pr')
            style_player.findChild(QLabel, "pr").setPixmap(QPixmap('./resources/style_pr.png'))
            style_player.show()
            btn3.setEnabled(False)
            btn2.setEnabled(True)
            btn2.show()

        content_img = QLabel()
        content_img.setScaledContents(True)

        content_cb = QComboBox()
        content_cb.currentIndexChanged[str].connect(next_content)
        content_cb2 = QComboBox()
        content_cb2.currentIndexChanged[str].connect(content_changed)

        content_player = MusicPlayer()
        content_player.set_song('./resources/content.midi')
        content_player.findChild(QPushButton, 'play').clicked.connect(lambda: (
            style_player.findChild(QPropertyAnimation).stop(), syn_player.findChild(QPropertyAnimation).stop()))

        style_img = QLabel()
        style_img.setScaledContents(True)
        style_cb = QComboBox()
        style_cb.currentIndexChanged[str].connect(next_style)
        style_player = MusicPlayer()
        style_player.set_song('./resources/style.midi')
        style_player.findChild(QPushButton, 'play').clicked.connect(lambda: (
            content_player.findChild(QPropertyAnimation).stop(), syn_player.findChild(QPropertyAnimation).stop()))

        syn_img = QLabel()
        syn_img.setScaledContents(True)

        btn2 = QPushButton("stylize")
        btn2.clicked.connect(stylize)
        syn_player = MusicPlayer()
        syn_player.set_song('./resources/syn.midi')
        syn_player.findChild(QPushButton, 'play').clicked.connect(lambda: (
            content_player.animation.stop(), style_player.animation.stop()))

        gl.addWidget(content_img, 0, 0)
        gl.addWidget(content_cb, 1, 0)
        gl.addWidget(content_cb2, 2, 0)
        gl.addWidget(content_player, 3, 0)

        gl.addWidget(style_img, 0, 1)
        gl.addWidget(style_cb, 1, 1)
        gl.addWidget(style_player, 3, 1)

        gl.addWidget(syn_img, 0, 2)
        gl.addWidget(btn2, 2, 2)
        gl.addWidget(syn_player, 3, 2)

        content_img.setMinimumSize(200, 200)
        content_cb.setMinimumWidth(200)
        content_cb2.setMinimumWidth(200)
        content_player.setMinimumSize(200, 200)

        style_img.setMinimumSize(200, 200)
        style_cb.setMinimumWidth(200)
        style_player.setMinimumSize(200, 200)

        syn_img.setMinimumSize(200, 200)
        btn2.setMinimumWidth(200)
        syn_player.setMinimumSize(200, 200)
        sp = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sp.setRetainSizeWhenHidden(True)
        syn_player.setSizePolicy(sp)

        gl.setColumnStretch(0, 1)
        gl.setColumnStretch(1, 1)
        gl.setColumnStretch(2, 1)

        gl.setRowStretch(0, 1)
        gl.setRowStretch(3, 1)

        def rs(s):
            if s.slider.value() != 0:
                s.slider.setValue(0)
                img_edit()

        s1 = MySliderLayout()
        s1.set_argu('Brightness', -70, 70)
        s1.slider.sliderReleased.connect(img_edit)
        s1.reset_btn.clicked.connect(lambda: rs(s1))

        s2 = MySliderLayout()
        s2.set_argu(' Contrast ', -70, 70)
        s2.slider.sliderReleased.connect(img_edit)
        s2.reset_btn.clicked.connect(lambda: rs(s2))

        btn3 = QPushButton("Enhance")
        btn3.clicked.connect(fine_tune)
        btn3.setEnabled(False)

        layout4.addLayout(s1)
        layout4.addLayout(s2)

        layout4.addWidget(btn3)

        content_cb.addItems(names)
        style_cb.addItems(style_names)


if __name__ == '__main__':


    app = QApplication(sys.argv)

    window = Window('Vis2Mus')
    window.show()
    sys.exit(app.exec_())