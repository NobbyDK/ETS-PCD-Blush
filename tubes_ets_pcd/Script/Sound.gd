extends Node

var click_sound: AudioStreamPlayer

func _ready():
	print("Sound.gd loaded!") # ✅ Debug 1
	click_sound = AudioStreamPlayer.new()
	add_child(click_sound)

	var stream = preload("res://assets/Sound/ClickSound.mp3")
	print(stream) # ✅ Debug 2 apakah null?
	
	click_sound.stream = stream
	click_sound.volume_db = 0

func play_click():
	print("✅ Sound triggered!") # ✅ Debug 3
	click_sound.stop()      # ✅ reset state dulu
	click_sound.play()      # ✅ play dulu agar streaming dimulai
	click_sound.seek(0.4)   # ✅ pastikan langsung mulai dari awal
