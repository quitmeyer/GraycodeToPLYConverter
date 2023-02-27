Package:        Editor Window Fullscreen (v1.3.1)
Publisher:      Crystal Conflux
Support:        Please post in the forum thread or email us if you have any problems, questions, or suggestions.
Forum Thread:   https://forum.unity3d.com/threads/473619/
Support Email:  support@crystalconflux.com

Key Features:   • Allows any editor window, including the in-editor game view, to be opened in fullscreen mode.
                • Supports Windows 10 and Mac OSX (macOS).
                • Now with improved support for fullscreening Timeline and Playmaker windows.
                • Option to auto-enter fullscreen on play, and exit fullscreen on game stop (and vice-versa).​
                • Supports multiple game camera displays tied to different physical displays.
                • Easily customizable hotkeys and settings, via a GUI settings window (shortcut Ctrl+Shift+F8).
                • Multi-display support. Each window type can be opened in fullscreen on one or more screens.
                • Supports per-monitor display scaling (hi-DPI).
                • Retains the Scene View state (lighting, world position etc.) when opening the Scene View in fullscreen.
                • Exits existing fullscreens on the same screen when opening a new one.
                • Ability to fullscreen the main Unity window. It returns to its original size when exiting fullscreen mode.
                • Cursor lock is retained when entering fullscreen game view.
                • Reloads previous fullscreen state when restarting Unity.
                • Ability to toggle the top toolbar for the game view, scene view, and main view when in fullscreen mode.​
                • Create additional hotkeys for fullscreen game views and customize their settings separately.
                • Choose the game display, aspect ratio, scale, and more, for each customized game view.
                • Switch between user-defined aspect ratios and game "Displays" at the touch of a button.

Tutorial Video: https://youtu.be/NXQioqLA0tI

Installation:   No special installation is required for fullscreen functionality. Once you've imported this package into your project, all of the fullscreen hotkeys should work out-of-the-box.
                It is recommended you leave the package installed in the root "Plugins" folder to allow for first-pass compilation and avoid the extension adding to your game's compile time.
                On Mac, The EWFMac.bundle must remain within the Plugins directory subfolders. If it is moved outside of the Plugins directory, Mac fullscreen functionality will be broken.

                Demo Installation: If you want to play the demo, which is an optional and brief 2D tutorial on how to use this extension, you must:
                  • Import the Standard Assets from the Asset Store, namely the "2D", "Cameras", and "CrossPlatformInput" folders of the Standard Assets.
                  • Under the "Demo" folder, open the scene "Demo_2D", and hit play. Since the demo is for demonstrating editor features, it will only work when run from within the editor. It will not be functional in standalone mode.

Defaults:       By default, fullscreen windows open at the current mouse position. This is adjustable in settings.
                When toggling fullscreen, a new fullscreen window is created if none of that type exists at the desired position. If one already exists at that position, it is closed.
                Default Hotkeys (Default Mac hotkey in brackets):
                  • Toggle fullscreen for the Main Unity Window: F8 (⌘F8)
                  • Toggle fullscreen for the Scene View: F10 (⌘F10)
                  • Toggle fullscreen for the Game Window: F11 (⌘F11)
                  • Toggle fullscreen for the focused window: F9 (⌘F9) (To focus a window, simply click on it)
                  • Toggle fullscreen for the window under cursor: Ctrl+F9 (⌘+⌥+F9)
                  • Show toolbar while in fullscreen: F12 (⌘F12)
                  • Close all fullscreen windows: Ctrl+F8 (⌘+⌥+F8)

Menu Items:     There is a menu-item for every fullscreen hotkey, located in the menu bar under the "Window" menu and the "Editor Window Fullscreen" submenu. 

Settings:       Settings and hotkeys can be changed in the graphical "Fullscreen Window Settings" window.
                You can access this through the Window menu, by going to "Window >> Editor Window Fullscreen >> Fullscreen Window Settings...".
                Alternatively you can use the hotkey Ctrl+Shift+F8 to immediately open the settings window.
                Here you can change:
                  • The hotkeys for creating and closing fullscreen editor windows.
                  • Whether to show a notification when entering fullscreen
                  • Options to link game start and stop events to exiting and entering of a fullscreen game window.
                  • The position where each window type will enter fullscreen. (At the current window position, mouse position, or a custom position). This determines which screen the fullscreen window will open on.
                  • Whether to show the top toolbar by default, when entering fullscreen mode. (Only applies to Scene View, Game View, and Main Window, which have top toolbars).
                  • The hotkey for showing/hiding the top toolbar, if one exists, when in fullscreen mode.
                  • The hotkey for closing all fullscreen windows.

                You can also create custom fullscreen game views:
                  • Create unlimited custom game views, each one with its own settings.
                  • Assign hotkeys to them and choose their aspect ratio, game display, scale, and more.

Code Usage:     EditorFullscreenController — If you would like to control fullscreen windows through your own code, you can easily do this by calling methods in the EditorFullscreenController class. 

                    1. First, add the following using statement to the top of your script:
                        using EditorWindowFullscreen;
                
                    2. Then, call your desired static method of EditorFullscreenController (More details about each method and their parameters can be found in the summary text of each method, within the script itself):
                      • To open fullscreens at positions set according to the current settings (All of these methods are already hotkeyed):
                          ToggleMainWindowFullscreen()        — Toggles fullscreen for the main editor window.
                          ToggleSceneViewFullscreen()         — Toggles fullscreen for the scene view.
                          ToggleGameViewFullscreen()          — Toggles fullscreen for the game view.
                          ToggleFocusedWindowFullscreen()     — Toggles fullscreen for the focused window.
                          ToggleWindowUnderCursorFullscreen() — Toggle fullscreen for the window under the cursor.
                          ToggleTopToolbar()                  — Toggles the top toolbar for the currently focused fullscreen window. (Only applies to Scene View, Game View, and Main Window, which have top toolbars).

                      • To find out if a window type is fullscreen:
                          WindowTypeIsFullscreen(Type windowType)                                       — Returns true if a window type is fullscreen on any screen.
                          WindowTypeIsFullscreenOnScreenAtPosition(Type windowType, Vector2 atPosition) — Returns true if a window type is fullscreen on the screen at the specified position.

                      • To create/destroy fullscreen windows:
                          ToggleFullscreenAtMousePosition(Type windowType, bool showTopToolbar)         — Toggle fullscreen at the current mouse position for the window with the specified type.
                          ToggleFullscreen(Type windowType, Vector2 atPosition, bool showTopToolbar)    — Toggle fullscreen for a window type, on the screen at a position.
                          ExitGameFullscreens(bool onlyThoseCreatedAtGameStart)                         — Exits the fullscreen game views. If the parameter is true, only exits the game views which were created when the game was started.
                          CloseAllEditorFullscreenWindows()                                             — Closes all fullscreen editor windows.

                EditorWindow Extension Methods — Existing editor windows can be opened in fullscreen by directly calling an extension method on any EditorWindow object:
                    
                    1. First, add the following using statement to the top of your script:
                         using EditorWindowFullscreen;

                    2. Now you may call your desired method on any EditorWindow of your choice (Including custom EditorWindows):
                        myEditorWindow.ToggleFullscreen()                                                         — Toggles fullscreen on or off for the EditorWindow at its current position.
                        myEditorWindow.ToggleFullscreen(Vector2 atPosition)                                       — Toggles fullscreen on or off for the EditorWindow, on the screen at the specified position.
                        myEditorWindow.SetFullscreen(bool setFullscreen, bool showTopToolbar)                     — Sets fullscreen to true or false for the EditorWindow at its current position, and shows/hides the toolbar.
                        myEditorWindow.SetFullscreen(bool setFullscreen, Vector2 atPosition, bool showTopToolbar) — Sets fullscreen to true or false for the EditorWindow, on the screen at the specified position, and shows/hides the toolbar.
                        myEditorWindow.IsFullscreen()                                                             — Returns true if the EditorWindow is currently fullscreen on its current screen.
                        myEditorWindow.IsFullscreen(Vector2 atPosition)                                           — Returns true if the EditorWindow is currently fullscreen at the specified position.
                        myEditorWindow.SetBorderlessPosition(Rect position)                                       — Make the EditorWindow borderless and give it an accurate position and size

Scripts:        Controller Folder - This folder contains the scripts which control the fullscreen state.
                  • EditorFullscreenController - This class controls the fullscreen state. You can call the methods of this class if you want to create custom fullscreen windows or your own fullscreen controller.

                Internals Folder - This folder contains the scripts which the controller depends upon to create fullscreen windows. Modifying these scripts is not recommended.