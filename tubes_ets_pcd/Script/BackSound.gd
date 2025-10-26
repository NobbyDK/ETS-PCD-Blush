# BackSound.gd
extends AudioStreamPlayer

func _ready():
	var stream = preload("res://assets/Sound/BackSound.mp3")
	if stream is AudioStream:
		stream.loop = true
	
	self.stream = stream
	self.volume_db = -6
	
	# Mulai musik otomatis
	play_music()

func play_music():
	if not playing:
		play()

func stop_music():
	if playing:
		stop()

func toggle_music():
	if playing:
		stop_music()
	else:
		play_music()

func is_music_playing() -> bool:
	return playing
