using UnityEngine;
using UnityEditor;
using System;
using System.Collections.Generic;
using EditorWindowFullscreen;

namespace EditorWindowFullscreenDemo
{
    public class EditorDemoController
    {
        private static List<EditorFullscreenSettings.FullscreenOption> hintOptions;
        private static Type lastFullscreenedWindowType;
        private static bool initialized = false;
        
        [RuntimeInitializeOnLoadMethod]
        public static void Initialize()
        {
            if (DemoCharacter.instance == null) return; //Only activate if playing the demo and the character has been initialized.
            if (!EditorApplication.isPlayingOrWillChangePlaymode && !EditorApplication.isPlaying) return; //Only activate if playing.
            DateTime startInit = DateTime.Now;

            EditorFullscreenState.FullscreenEventHandler += OnFullscreenEvent;
            EditorFullscreenController.FullscreenHotkeyEventHandler += OnFullscreenHotkeyEvent;
#if UNITY_2017_2_OR_NEWER
            EditorApplication.playModeStateChanged += DisableController;
#else
            EditorApplication.playmodeStateChanged += DisableController;
#endif

            initialized = true;

            var settings = EditorFullscreenSettings.settings;
            var hints = new List<string>();
            var displays = EditorDisplay.GetAllDisplays();
            hintOptions = new List<EditorFullscreenSettings.FullscreenOption>();

            AddHint(hints, settings.mainUnityWindow, "Press {0} to fullscreen the Main Window \n({1})");
            AddHint(hints, settings.gameWindow, "Press {0} to fullscreen the Game View \n({1})");

            if (displays.Count > 1 && settings.sceneWindow.openAtPosition == EditorFullscreenSettings.OpenFullscreenAtPosition.AtMousePosition)
                AddHint(hints, settings.sceneWindow, hints[1] + "\n\nPress {0} to open a fullscreen Scene View on ANOTHER screen\n(opens {1})");
            else
                AddHint(hints, settings.sceneWindow, "Press {0} to fullscreen the Scene View \n({1})");

            AddHint(hints, settings.toggleTopToolbar, "Press {0} to toggle the top toolbar for the Game View \n(make sure to focus the Game View first)");

            AddHint(hints, settings.closeAllFullscreenWindows, ""); //Close the (window over main unity view, if one exists), by pushing {x}.)

            AddHint(hints, settings.currentlyFocusedWindow, "Click inside the Inspector window/tab in the Editor.\nThen press {0} to fullscreen it {1}.\n(This hotkey fullscreens the currently focused window).");
            AddHint(hints, settings.windowUnderCursor, "Hover the mouse over the Console window/tab in the Editor.\nThen press {0} to fullscreen it {1}.\n(This hotkey fullscreens the window under the cursor).");
            AddHint(hints, settings.closeAllFullscreenWindows, "Press {0} to close all fullscreen windows\n(First, make sure you have at least one fullscreen window open)");

            AddHint(hints, settings.closeAllFullscreenWindows, "Open the fullscreen settings window.\nUnder the Window menu, go to:\n\"Editor Window Fullscreen >> Fullscreen Window Settings...\"");

            AddHint(hints, settings.closeAllFullscreenWindows, "Demo complete!");

            DemoCharacter.hints = hints.ToArray();
            if (EWFDebugging.Enabled && EWFDebugging.enableTimerLogging) Debug.Log("Demo controller initialized. Took " + (DateTime.Now - startInit).Milliseconds + "ms.");
        }

#if UNITY_2017_2_OR_NEWER
        private static void DisableController(PlayModeStateChange state)
        {
            DisableController();
        }
#endif

        public static void DisableController()
        {
            if (initialized && !EditorApplication.isPlaying)
            {
                EditorFullscreenState.FullscreenEventHandler -= OnFullscreenEvent;
                EditorFullscreenController.FullscreenHotkeyEventHandler -= OnFullscreenHotkeyEvent;
#if UNITY_2017_2_OR_NEWER
                EditorApplication.playModeStateChanged -= DisableController;
#else
                EditorApplication.playmodeStateChanged -= DisableController;
#endif
                initialized = false;
            }
        }

        private static void AddHint(List<string> hints, EditorFullscreenSettings.FullscreenOption fullscreenOption, string format)
        {
            string openAtPos, keysDownString;
            GetInfoForFullscreenOption(fullscreenOption, out openAtPos, out keysDownString);
            var hint = string.Format(format, keysDownString, openAtPos);
            hints.Add(hint);
            hintOptions.Add(fullscreenOption);
        }

        private static void GetInfoForFullscreenOption(EditorFullscreenSettings.FullscreenOption fullscreenOption, out string openAtPos, out string keysDownString)
        {
            if (fullscreenOption.openAtPosition == EditorFullscreenSettings.OpenFullscreenAtPosition.AtSpecifiedPosition)
            {
                openAtPos = "at the position: " + fullscreenOption.position;
            }
            else
            {
                openAtPos = EditorFullscreenSettings.FormatCamelCaseName(fullscreenOption.openAtPosition.ToString()).ToLower().Replace("at ", "at the ");
            }
            keysDownString = EditorInput.GetKeysDownString(fullscreenOption.hotkey, fullscreenOption.modifiers);
        }

        static void OnFullscreenEvent(object window, Type windowType, Vector2 atPosition, bool enteredFullscreen)
        {
            if (enteredFullscreen) lastFullscreenedWindowType = windowType;
            bool toggledTopToolbar = false;
            if (window != null || windowType == EditorFullscreenState.MainWindowType)
            {
                EditorFullscreenState.WindowFullscreenState state = null;
                if (windowType == EditorFullscreenState.MainWindowType)
                    state = EditorFullscreenState.FindWindowState(null, windowType);
                else if (window != null)
                    state = EditorFullscreenState.FindWindowState((EditorWindow)window);

                if (state != null)
                {
                    var fullscreenOps = state.FullscreenOptions;
                    toggledTopToolbar = fullscreenOps != null && state.ShowTopToolbar != fullscreenOps.showToolbarByDefault;
                }
            }

            var character = DemoCharacter.instance;

            bool completedRoom = false;
            int currentRoom = character.enteredRooms.Count;
            switch (currentRoom)
            {
                case 1:
                    if (windowType == EditorFullscreenState.MainWindowType && enteredFullscreen) completedRoom = true;
                    break;
                case 2:
                    if (windowType == EditorFullscreenState.GameViewType && enteredFullscreen) completedRoom = true;
                    break;
                case 3:
                    if (DemoCharacter.hints[2].Contains("ANOTHER"))
                    {
                        //Must have game view fullscreen and open fullscreens scene view on another screen.
                        bool gameViewIsFullscreen = EditorFullscreenController.WindowTypeIsFullscreen(EditorFullscreenState.GameViewType);
                        bool sceneViewIsFullscreen = EditorFullscreenController.WindowTypeIsFullscreen(EditorFullscreenState.SceneViewType);
                        if (enteredFullscreen && gameViewIsFullscreen && sceneViewIsFullscreen && (windowType == EditorFullscreenState.GameViewType || windowType == EditorFullscreenState.SceneViewType))
                        {
                            completedRoom = true;
                        }
                    }
                    else
                    {
                        //Only have to open the scene view.
                        if (windowType == EditorFullscreenState.SceneViewType && enteredFullscreen) completedRoom = true;
                    }
                    break;
                case 4:
                    if (windowType == EditorFullscreenState.GameViewType && toggledTopToolbar)
                    {
                        completedRoom = true;
                    }
                    if (completedRoom || DemoCharacter.completedRooms == 4) {
                        if (CompletedHint5() == true) character.CompleteRoom(5, true);
                    }
                    break;
                case 5:
                    if (CompletedHint5() == true) completedRoom = true; //Must have closed the window covering the main window.
                    break;
                case 8:
                    if (enteredFullscreen && windowType == null) //windowType is null when closing all fullscreen editor windows. In this case enteredFullscreen is true if at least one fullscreen window was closed.
                    {
                        completedRoom = true;
                    }
                    break;
                case 9:
                    if (!enteredFullscreen && windowType == typeof(EditorFullscreenSettingsWindow))
                    {
                        completedRoom = true;
                    }
                    break;
            }

            if (completedRoom) character.CompleteRoom(currentRoom);
        }

        static void OnFullscreenHotkeyEvent(KeyCode keyCode, EventModifiers modifiers, bool setFullscreen)
        {
            var character = DemoCharacter.instance;
            bool completedRoom = false;
            int currentRoom = character.enteredRooms.Count;
            switch (currentRoom)
            {
                case 6:
                    var focusedWinOps = EditorFullscreenSettings.settings.currentlyFocusedWindow;
                    if (setFullscreen && EditorInput.HotkeysMatch(keyCode, modifiers, focusedWinOps.hotkey, focusedWinOps.modifiers))
                    {
                        if (lastFullscreenedWindowType == EditorFullscreenState.InspectorWindowType)
                            completedRoom = true;
                    }
                    break;
                case 7:
                    var winUnderCursorOps = EditorFullscreenSettings.settings.windowUnderCursor;
                    if (setFullscreen && EditorInput.HotkeysMatch(keyCode, modifiers, winUnderCursorOps.hotkey, winUnderCursorOps.modifiers))
                    {
                        if (lastFullscreenedWindowType == EditorFullscreenState.ConsoleWindowType)
                            completedRoom = true;
                    }
                    break;
            }
            if (completedRoom) character.CompleteRoom(currentRoom);
        }

        /// <summary>
        /// Updates the 5th hint. Returns true if the 5th room objective is complete.
        /// </summary>
        /// <returns></returns>
        private static bool CompletedHint5()
        {
            bool completedObjective = true;
            DemoCharacter.hints[4] = "";
            foreach (var state in EditorFullscreenState.fullscreenState.window)
            {
                if (state.IsFullscreen && state.WindowType != EditorFullscreenState.MainWindowType && EditorDisplay.ClosestToPoint(state.FullscreenAtPosition).Bounds.Contains(EditorMainWindow.position.center))
                {
                    completedObjective = false;
                    var fullscreenOption = state.FullscreenOptions;
                    string openAtPos, keysDownString;
                    GetInfoForFullscreenOption(fullscreenOption, out openAtPos, out keysDownString);

                    var windowName = state.WindowType.ToString();
                    if (windowName.Contains(".")) windowName = windowName.Split(".".ToCharArray())[1];
                    windowName = EditorFullscreenSettings.FormatCamelCaseName(windowName);

                    DemoCharacter.hints[4] += "Press " + keysDownString + " to close the " + windowName;
                    if (fullscreenOption.openAtPosition == EditorFullscreenSettings.OpenFullscreenAtPosition.AtMousePosition)
                        DemoCharacter.hints[4] += "\n(mouse must be hovering over the window)";
                    DemoCharacter.hints[4] += "\n";
                }
            }
            return completedObjective;
        }
    }
}