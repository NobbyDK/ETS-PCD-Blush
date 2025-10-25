extends Control


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
	
func _on_BtnExit_pressed():
	get_tree().quit()

func _on_start_pressed():
	get_tree().change_scene_to_file("res://scene/webcam_client_udp.tscn")

func _on_about_pressed():
	get_tree().change_scene_to_file("res://scene/about.tscn")

func _on_tutor_pressed() -> void:
	get_tree().change_scene_to_file("res://scene/tutor.tscn")

func _on_kredit_pressed() -> void:
	get_tree().change_scene_to_file("res://scene/kredit.tscn")


func _on_home_pressed() -> void:
	get_tree().change_scene_to_file("res://scene/menu_utama.tscn")
