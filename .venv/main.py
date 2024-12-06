import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import fftconvolve, butter, lfilter, resample
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QFileDialog, QMessageBox, QLabel, QSlider, \
    QProgressBar, QComboBox, QLineEdit
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
import sqlite3


class AudioPlayer(QThread):
    finished_signal = pyqtSignal(float)  # сигнал завершения воспроизведения с текущим временем

    def __init__(self, audio_data, samplerate):
        super().__init__()
        self.audio_data = audio_data
        self.samplerate = samplerate
        self.is_playing = False
        self.is_paused = False
        self.stream = False
        self.current_frame = 0  # текущий семпл
        self.block_size = 66150  # размер блока в семплах
        self.duration = len(audio_data) / samplerate

        self.reverb_intensity = 0
        self.chorus_intensity = 0.0
        self.eq_settings = {"low": 1.0, "mid": 1.0, "high": 1.0}
        self.delay_time = 0
        self.decay = 0

    def audio_callback(self, outdata, frames, time, status):  # обработчик блоков для последующей передачи в поток
        if self.is_paused or self.current_frame >= len(self.audio_data):
            outdata.fill(0)  # Пауза или конец данных - тишина
            return

        end_frame = min(self.current_frame + frames, len(self.audio_data))
        block = self.audio_data[self.current_frame:end_frame]

        # перевод сигнала в стереофонический
        if block.ndim == 1:  # моно
            block = block.reshape(-1, 1)
        elif block.ndim == 2 and block.shape[1] == 2:  # стерео
            pass
        else:
            raise ValueError(f"Unsupported audio format with shape {block.shape}")

        if self.reverb_intensity:
            block = self.apply_reverb(block)
        if self.chorus_intensity:
            block = self.apply_chorus(block)
        block = self.apply_equalizer(block)
        if self.delay_time:
            block = self.apply_delay(block)

        # если block имеет меньше семплов, чем должно быть, заполняем остаток нулями
        if len(block) < frames:
            block = np.pad(block, ((0, frames - len(block)), (0, 0)), 'constant')

        # заполняем буфер потока
        outdata[:] = block
        self.current_frame += frames

    def run(self):  # запуск воспроизведения
        self.is_playing = True
        self.is_paused = False
        self.current_frame = 0

        # поток воспроизведения
        with sd.OutputStream(
                samplerate=self.samplerate,
                blocksize=self.block_size,
                channels=2,
                dtype="float32",
                callback=self.audio_callback,
        ) as self.stream:
            while self.is_playing:
                if self.is_paused:
                    self.msleep(100)  # ожидание, если на паузе

        self.is_playing = False
        self.finished_signal.emit(self.current_frame / self.samplerate)  # закрытие потока

    def play(self):  # воспроизведение
        if self.is_playing and not self.is_paused:  # если воспроизведение активно
            return

        if not self.stream:  # если поток еще не создан
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.audio_data.shape[1] if len(self.audio_data.shape) > 1 else 1,
                callback=self.audio_callback,
                finished_callback=self.on_stop
            )
        self.is_playing = True
        self.is_paused = False
        self.stream.start()

    def pause(self):  # пауза
        if self.stream and self.is_playing:
            self.is_paused = not self.is_paused

    def stop(self):  # сброс
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0

    def apply_reverb(self, signal):  # применение реверберации
        if self.reverb_intensity == 0.0:
            return signal

        # создание импульсного отклика(эффекта затухания) на 0.5 с для каждого семпла
        impulse_response = np.zeros(int(self.samplerate * 0.5))
        impulse_response[0] = 1.0
        for i in range(1, len(impulse_response)):
            impulse_response[i] = impulse_response[i - 1] * (1.0 - self.reverb_intensity)

        # применяем реверберацию ко всем каналам сигнала
        if signal.ndim == 2:  # стерео
            processed_signal = np.zeros_like(signal)
            for channel in range(signal.shape[1]):
                processed_signal[:, channel] = fftconvolve(signal[:, channel], impulse_response, mode="full")[
                                               : len(signal)]
        else:  # моно
            processed_signal = fftconvolve(signal, impulse_response, mode="full")[: len(signal)]
            # наложение импульсного отклика на первоначальный сигнал

        return processed_signal

    def apply_equalizer(self, signal):  # применение фильтров эквалайзера
        low_gain = self.eq_settings["low"]
        mid_gain = self.eq_settings["mid"]
        high_gain = self.eq_settings["high"]

        low = self.band_filter(signal, 20, 250, gain=low_gain)
        mid = self.band_filter(signal, 250, 4000, gain=mid_gain)
        high = self.band_filter(signal, 4000, 20000, gain=high_gain)

        return low + mid + high

    def band_filter(self, signal, lowcut, highcut, gain):  # фильтр определённой полосы частот
        nyquist = 0.5 * self.samplerate  # частота Найквиста
        low = lowcut / nyquist  # перевод частоты в коэффициент
        high = highcut / nyquist
        a, b = butter(2, [low, high], btype='band')  # коэффициенты для фильтрации частот
        filtered_signal = lfilter(a, b, signal, axis=0)  # фильтрация частот в полосовом фильтре
        return filtered_signal * gain

    def apply_chorus(self, signal):  # применение эффекта хора
        delay_max = int(self.samplerate * 0.03)  # максимальная задержка 30 мс
        delay_min = int(self.samplerate * 0.01)  # минимальная задержка 10 мс
        depth = int(self.chorus_intensity * delay_max)  # глубина модуляции

        # создание плавно модулирующей задержки при помощи синусоиды
        modulation = (np.sin(2 * np.pi * np.arange(len(signal)) * 0.1 / self.samplerate) + 1) / 2
        delays = (modulation * (delay_max - delay_min) + delay_min).astype(int)

        chorus_signal = np.zeros_like(signal)
        for i in range(len(signal)):
            delay = delays[i]
            if i >= delay:
                chorus_signal[i] = signal[i] + signal[i - delay] * 0.7  # Задержка и ослабление амплитуды

        return (signal + chorus_signal) / 2  # накладываем эффект на оригинал

    def apply_delay(self, signal):  # применение дилея
        delay_samples = int(self.delay_time * self.samplerate)
        delayed_signal = np.pad(signal, ((delay_samples, 0), (0, 0)), mode='constant')[:-delay_samples]
        delayed_signal *= self.decay  # затухание повторяющегося сигнала
        return signal + delayed_signal  # смешивание оригинала с задержанным сигналом


class AudioApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Advanced music player')
        self.initUI()
        self.audio_data = None
        self.samplerate = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.player = None

    def initUI(self):
        self.setGeometry(200, 200, 640, 360)
        self.msg = QMessageBox()
        self.file_path_button = QPushButton('Open audio', self)
        self.play_button = QPushButton('Play', self)
        self.stop_button = QPushButton('Stop', self)

        self.file_path_button.clicked.connect(self.getDirectory)
        self.play_button.clicked.connect(self.play_audio)
        self.stop_button.clicked.connect(self.stop_audio)

        self.play_button.move(70, 170)
        self.stop_button.move(70, 220)
        self.file_path_button.move(70, 270)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.move(30, 130)
        self.progress_bar.resize(250, 15)

        self.time_label = QLabel("00:00 / 00:00", self)
        self.time_label.move(30, 120)

        self.reverb_main_label = QLabel('Reverberation:', self)
        self.reverb_main_label.move(235, 45)
        self.reverb_main_label.setStyleSheet("font-size: 22px; font-style: italic")
        self.reverb_label = QLabel("Reverb Intensity: 0%", self)
        self.reverb_label.resize(150, 30)
        self.reverb_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.reverb_slider.setMinimum(0)
        self.reverb_slider.setMaximum(100)
        self.reverb_slider.setValue(0)
        self.reverb_slider.valueChanged.connect(self.update_reverb_intensity)
        self.reverb_slider.move(240, 80)
        self.reverb_label.move(240, 100)
        self.reset_reverb_button = QPushButton('reset reverb', self)
        self.reset_reverb_button.clicked.connect(self.reset_reverb)
        self.reset_reverb_button.move(235, 125)
        self.reset_reverb_button.resize(80, 25)

        self.equalizer_main_label = QLabel('Equalizer:', self)
        self.equalizer_main_label.move(235, 160)
        self.equalizer_main_label.setStyleSheet("font-size: 22px; font-style: italic")
        self.equalizer_sliders = {}
        # подобное создание слайдеров удобно, если нужно будет  добавить больше частот для эквализации
        for band in ["low", "mid", "high"]:
            slider_freq_label = QLabel(f"{band.capitalize()}", self)
            slider_equal_label = QLabel('0 dB', self)
            slider_equal_label.resize(150, 30)

            slider = QSlider(Qt.Orientation.Vertical, self)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.valueChanged.connect(lambda value, b=band, l=slider_equal_label: self.update_eq(value, b, l))
            slider.move(240 + 50 * ["low", "mid", "high"].index(band), 200)
            slider_freq_label.move(240 + 50 * ["low", "mid", "high"].index(band), 290)
            slider_equal_label.move(240 + 50 * ["low", "mid", "high"].index(band), 300)
            self.equalizer_sliders[band] = slider
        self.reset_eq_button = QPushButton('reset eq', self)
        self.reset_eq_button.clicked.connect(self.reset_eq)
        self.reset_eq_button.move(235, 325)
        self.reset_eq_button.resize(80, 25)

        self.chorus_main_label = QLabel('Chorus:', self)
        self.chorus_main_label.move(395, 45)
        self.chorus_main_label.setStyleSheet("font-size: 22px; font-style: italic")
        self.chorus_label = QLabel("Chorus Intensity: 0%", self)
        self.chorus_label.resize(150, 30)
        self.chorus_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.chorus_slider.setMinimum(0)
        self.chorus_slider.setMaximum(100)
        self.chorus_slider.setValue(0)
        self.chorus_slider.valueChanged.connect(self.update_chorus_intensity)
        self.chorus_slider.move(400, 80)
        self.chorus_label.move(400, 100)
        self.reset_chorus_button = QPushButton('reset chorus', self)
        self.reset_chorus_button.clicked.connect(self.reset_chorus)
        self.reset_chorus_button.move(395, 125)
        self.reset_chorus_button.resize(80, 25)

        self.pitch_main_label = QLabel('Delay:', self)
        self.pitch_main_label.move(535, 45)
        self.pitch_main_label.setStyleSheet("font-size: 22px; font-style: italic")
        self.delay_label = QLabel("Delay Time: 0 s", self)  # время дилея
        self.delay_label.resize(150, 30)
        self.delay_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.delay_slider.setMinimum(0)
        self.delay_slider.setMaximum(1000)
        self.delay_slider.setValue(0)
        self.delay_slider.valueChanged.connect(self.update_delay_time)
        self.delay_slider.move(540, 80)
        self.delay_label.move(540, 100)
        self.reset_delay_button = QPushButton('reset delay', self)
        self.reset_delay_button.clicked.connect(self.reset_delay)
        self.reset_delay_button.move(535, 185)
        self.reset_delay_button.resize(80, 25)

        self.decay_label = QLabel("Decay: 0", self)  # затухание дилея
        self.decay_label.resize(150, 30)
        self.decay_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.decay_slider.setMinimum(0)
        self.decay_slider.setMaximum(100)
        self.decay_slider.setValue(50)
        self.decay_slider.valueChanged.connect(self.update_decay)
        self.decay_slider.move(540, 140)
        self.decay_label.move(540, 160)

        self.preset_list = QComboBox(self)
        self.preset_list.move(400, 10)
        self.preset_list.resize(200, 30)
        self.preset_list.currentIndexChanged.connect(self.load_preset)

        # Кнопки для управления пресетами
        self.save_preset_name = QLineEdit(self)
        self.save_preset_button = QPushButton('Save Preset', self)
        self.delete_preset_button = QPushButton('Delete Preset', self)

        self.save_preset_button.clicked.connect(self.save_preset)
        self.delete_preset_button.clicked.connect(self.delete_preset)

        self.save_preset_name.move(50, 10)
        self.save_preset_button.move(200, 10)
        self.delete_preset_button.move(300, 10)

        self.update_presets_list()

    def update_presets_list(self):  # обновляет список пресетов
        self.preset_list.clear()
        conn = sqlite3.connect("presets.db")
        c = conn.cursor()

        c.execute('SELECT name FROM presets WHERE is_builtin = 1')
        built_in_presets = [row[0] for row in c.fetchall()]
        if built_in_presets:
            self.preset_list.addItem("-- Built-in Presets --")
            self.preset_list.addItems(built_in_presets)
        print(3)
        c.execute('SELECT name FROM presets WHERE is_builtin = 0')
        user_presets = [row[0] for row in c.fetchall()]
        if user_presets:
            self.preset_list.addItem("-- User Presets --")
            self.preset_list.addItems(user_presets)

        conn.close()

    def load_preset(self):  # загрузка выбранного пресета из выпадающего списка
        preset_name = self.preset_list.currentText()
        if preset_name.startswith("--"):
            return

        conn = sqlite3.connect("presets.db")
        c = conn.cursor()
        c.execute('''SELECT * FROM presets WHERE name = ?''', (preset_name,))
        preset = c.fetchone()

        conn.close()
        if preset:
            preset = {
                "name": preset[1],
                "reverb": preset[2],
                "chorus": preset[3],
                "low_freq": preset[4],
                "mid_freq": preset[5],
                "high_freq": preset[6],
                "delay": preset[7],
                "decay": preset[8]
            }
        if preset:
            self.apply_preset(preset)

    def save_preset(self):  # сохранение текущих настроек эффектов как новый пресет
        if not self.save_preset_name.text():
            self.msg.setText('Fill in the field with the preset name')
            self.msg.setIcon(QMessageBox.Icon.Warning)
            self.msg.exec()
            return
        conn = sqlite3.connect("presets.db")
        c = conn.cursor()
        c.execute('SELECT name FROM presets')
        result = [row[0] for row in c.fetchall()]
        if self.save_preset_name.text() in result:
            self.msg.setText('Preset with that name already exists')
            self.msg.setIcon(QMessageBox.Icon.Warning)
            self.msg.exec()
            return
        conn.close()
        reverb = self.reverb_slider.value() / 100
        chorus = self.chorus_slider.value() / 100
        low_freq = self.equalizer_sliders["low"].value() / 100 + 1
        mid_freq = self.equalizer_sliders["mid"].value() / 100 + 1
        high_freq = self.equalizer_sliders["high"].value() / 100 + 1
        delay = self.delay_slider.value() / 1000
        decay = self.decay_slider.value() / 100

        conn = sqlite3.connect("presets.db")
        c = conn.cursor()
        # Сохраняем новый пользовательский пресет в базу данных (is_builtin = 0)
        c.execute('''INSERT INTO presets (name, reverb, chorus, low_freq, mid_freq, high_freq, delay, decay, is_builtin)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)''',
                  (self.save_preset_name.text(), reverb, chorus, low_freq, mid_freq, high_freq, delay, decay))
        conn.commit()
        conn.close()
        print(1)
        self.update_presets_list()

    def delete_preset(self):  # удаляет выбранный пользовательский пресет
        preset_name = self.preset_list.currentText()

        # Пропускаем встроенные пресеты
        if preset_name.startswith("--") or self.is_builtin_preset(preset_name):
            self.msg.setText('Cannot delete built-in presets')
            self.msg.setIcon(QMessageBox.Icon.Warning)
            self.msg.exec()
            return

        conn = sqlite3.connect("presets.db")
        c = conn.cursor()
        c.execute('''DELETE FROM presets WHERE name = ? AND is_builtin = 0''', (preset_name,))
        conn.commit()
        conn.close()
        self.update_presets_list()

    def apply_preset(self, preset):  # применение настроек из выбранного пресета
        self.reverb_slider.setValue(int(preset["reverb"] * 100))
        self.chorus_slider.setValue(int(preset["chorus"] * 100))
        self.equalizer_sliders["low"].setValue(int((preset["low_freq"] - 1) * 100))
        self.equalizer_sliders["mid"].setValue(int((preset["mid_freq"] - 1) * 100))
        self.equalizer_sliders["high"].setValue(int((preset["high_freq"] - 1) * 100))
        self.delay_slider.setValue(int(preset["delay"] * 1000))
        self.decay_slider.setValue(int(preset["decay"] * 100))

    def is_builtin_preset(self, preset_name):  # встроенный ли эффект
        conn = sqlite3.connect("presets.db")
        c = conn.cursor()
        c.execute('SELECT is_builtin FROM presets WHERE name = ?', (preset_name,))
        result = c.fetchone()
        conn.close()
        return result and result[0] == 1

    def getDirectory(self):  # окно выбора файла
        path = QFileDialog.getOpenFileName(self, 'Open audio', 'c:\\', "Audio files (*.wav)")
        print(path)
        if path[0][-4:] == '.wav':
            self.download_audio(path[0])
        else:
            self.msg.setText('Incorrect format')
            self.msg.setIcon(QMessageBox.Icon.Warning)
            self.msg.exec()

    def download_audio(self, path):  # загрузка аудио-дорожки и частоты дискретизации из файла
        self.audio_data, self.samplerate = sf.read(path)
        self.player = AudioPlayer(self.audio_data, self.samplerate)  # плеер
        print('downloaded')

    def play_audio(self):  # запуск, продолжение или пауза воспроизведения
        if self.audio_data is not None:
            if not self.player.is_playing or self.player.is_paused:
                if not self.player.is_playing:
                    self.player.start()
                    self.timer.start(100)
                    self.player.is_playing = True
                    self.player.is_paused = False
                    self.play_button.setText('Play' if not self.player.is_playing or self.player.is_paused else 'Pause')
                elif self.player.is_paused:
                    self.player.play()
            else:
                self.player.pause()
            self.play_button.setText('Play' if not self.player.is_playing or self.player.is_paused else 'Pause')
        else:
            self.msg.setText('Audio is not downloaded')
            self.msg.setIcon(QMessageBox.Icon.Warning)
            self.msg.exec()

    def stop_audio(self):  # сброс воспроизведения
        if self.player and self.player.stream:
            self.player.stop()
            self.player.is_playing = False
            self.player.is_paused = False
            self.play_button.setText('Play' if not self.player.is_playing or self.player.is_paused else 'Pause')
        else:
            self.msg.setText('Audio is not playing')
            self.msg.setIcon(QMessageBox.Icon.Warning)
            self.msg.exec()

    def update_progress(self):
        if self.player:
            if self.player.current_frame <= len(self.audio_data):
                current_time = self.player.current_frame / self.samplerate
            else:
                current_time = len(self.audio_data) / self.samplerate
            duration = len(self.audio_data) / self.samplerate
            self.progress_bar.setValue(int((current_time / duration) * 100))
            self.time_label.setText(f"{self.format_time(current_time)} / {self.format_time(duration)}")

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02}:{seconds:02}"

    def update_reverb_intensity(self, value):  # обновление интенсивности ревербации в зависимости от слайдера
        intensity = value / 100
        self.reverb_label.setText(f"Reverb Intensity: {value}%")
        if self.player:
            self.player.reverb_intensity = intensity

    def reset_reverb(self):
        self.reverb_slider.setValue(0)

    def reset_eq(self):
        for band in ["low", "mid", "high"]:
            slider = self.equalizer_sliders[band]
            slider.setValue(0)

    def update_eq(self, value, band, label):  # обновление значений эквалайзера
        label.setText(f"{value / 10} dB")
        if self.player:
            self.player.eq_settings[band] = 10 ** (value / 200)  # преобразование dB в коэффициент

    def update_chorus_intensity(self, value):
        intensity = value / 100.0
        self.chorus_label.setText(f"Chorus Intensity: {value}%")
        if self.player:
            self.player.chorus_intensity = intensity

    def reset_chorus(self):
        self.chorus_slider.setValue(0)

    def update_delay_time(self, value):  # обновление времени задержки
        delay_time = value / 1000  # преобразуем в секунды
        self.delay_label.setText(f"Delay Time: {delay_time:.1f} s")
        if self.player:
            self.player.delay_time = delay_time

    def update_decay(self, value):  # обновление коэффициента затухания
        decay = value / 100
        self.decay_label.setText(f"Decay: {decay:.2f}")
        if self.player:
            self.player.decay = decay

    def reset_delay(self):
        self.delay_slider.setValue(0)
        self.decay_slider.setValue(0)


app = QApplication(sys.argv)
window = AudioApp()
window.show()
sys.exit(app.exec())
