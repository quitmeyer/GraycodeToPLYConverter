/* 
 * Author:  Johanan Round
 * Package: Editor Window Fullscreen
 * License: Unity Asset Store EULA (Editor extension asset. Requires 1 license per machine.)
 */

using UnityEngine;
using UnityEditor;
using System;
using System.Collections.Generic;
using FullscreenOption = EditorWindowFullscreen.EditorFullscreenSettings.FullscreenOption;

namespace EditorWindowFullscreen
{
    /// <summary>
    /// Purpose: Controls the current fullscreen state.
    /// </summary>
    public class EditorFullscreenController
    {
        private static EditorFullscreenSettings settings
        {
            get { return EditorFullscreenSettings.settings; }
        }
        internal delegate void FullscreenHotkeyEvent(KeyCode keyCode, EventModifiers modifiers, bool setFullscreen);
        internal static event FullscreenHotkeyEvent FullscreenHotkeyEventHandler;

        internal static FullscreenOption triggeredHotkey;

        /******************************************/
        /************ Hotkeyed Methods ************/
        /******************************************/

        /// <summary>
        /// Toggles fullscreen for the main editor window.
        /// </summary>
        ///
        public static bool ToggleMainWindowFullscreen()
        {
            return EditorFullscreenState.ToggleFullscreenUsingOptions(EditorFullscreenState.MainWindowType, settings.mainUnityWindow);
        }

        /// <summary>
        /// Toggles fullscreen for the scene view.
        /// </summary>
        public static bool ToggleSceneViewFullscreen()
        {
            return EditorFullscreenState.ToggleFullscreenUsingOptions(typeof(CustomSceneView), settings.sceneWindow);
        }

        /// <summary>
        /// Toggles fullscreen for the game view. (Using options for the primary Game Window).
        /// </summary>
        public static void ToggleGameViewFullscreen()
        {
            ToggleGameViewFullscreen(false, settings.gameWindow.OptionID);
        }
        private static bool ToggleGameViewFullscreen(bool triggeredOnGameStart, int optionID)
        {
            EditorWindow focusedWindow = null;
            List<FullscreenOption> allGameWins = null;
            EditorFullscreenState.WindowFullscreenState state = null;
            bool setFullscreen;
            if (triggeredOnGameStart)
            {
                allGameWins = settings.AllGameWindows;
                setFullscreen = true;
            }
            else
            {
                setFullscreen = !EditorFullscreenState.WindowTypeIsFullscreenAtOptionsSpecifiedPosition(EditorFullscreenState.GameViewType, settings.GetFullscreenOption(optionID));
            }

            EditorFullscreenState.RunOnLoad methodToRun;
            if (!triggeredOnGameStart) methodToRun = () => ToggleGameViewFullscreen(false, optionID);
            else methodToRun = () => ToggleGameViewFullscreen(true, optionID);
            if (EditorFullscreenState.RunAfterInitialStateLoaded(methodToRun)) return setFullscreen;

            if (triggeredOnGameStart)
            {
                for (int i = 0; i < allGameWins.Count; i++)
                {
                    if (allGameWins[i].openOnGameStart)
                    {
                        if (!EditorFullscreenState.WindowTypeIsFullscreenAtOptionsSpecifiedPosition(EditorFullscreenState.GameViewType, allGameWins[i]))
                        {
                            EditorFullscreenState.ToggleFullscreenUsingOptions(null, EditorFullscreenState.GameViewType, allGameWins[i], triggeredOnGameStart, false);
                        }
                    }
                }
            }
            else
            {
                state = EditorFullscreenState.ToggleFullscreenUsingOptions(null, EditorFullscreenState.GameViewType, settings.GetFullscreenOption(optionID), triggeredOnGameStart, false, out setFullscreen);
                focusedWindow = EditorWindow.focusedWindow;
            }

            EditorMainWindow.Focus();
            if (focusedWindow != null) focusedWindow.Focus();

            if (!triggeredOnGameStart)
            {
                bool isPlaying = EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode;
                if (settings.startGameWhenEnteringFullscreen && !isPlaying && setFullscreen)
                {
                    //Enter play mode
                    EditorApplication.ExecuteMenuItem("Edit/Play");
                }
                else if (settings.stopGameWhenExitingFullscreen != EditorFullscreenSettings.StopGameWhenExitingFullscreen.Never && isPlaying && !setFullscreen)
                {
                    if (settings.stopGameWhenExitingFullscreen == EditorFullscreenSettings.StopGameWhenExitingFullscreen.WhenAnyFullscreenGameViewIsExited || !WindowTypeIsFullscreen(EditorFullscreenState.GameViewType, state))
                    {
                        //Exit play mode
                        EditorApplication.ExecuteMenuItem("Edit/Play");
                    }
                }
            }
            return setFullscreen;
        }

        /// <summary>
        /// Toggles fullscreen for the focused window.
        /// </summary>
        /// <returns>True if the window became fullscreen. False if fullscreen was exited.</returns>
        public static bool ToggleFocusedWindowFullscreen()
        {
            if (EditorWindow.focusedWindow != null)
            {
                return EditorFullscreenState.ToggleFullscreenUsingOptions(EditorWindow.focusedWindow, EditorWindow.focusedWindow.GetType(), EditorFullscreenSettings.settings.currentlyFocusedWindow, false, true);
            }
            else return false;
        }

        /// <summary>
        /// Toggle fullscreen for the window under the cursor.
        /// </summary>
        /// <returns>True if the window became fullscreen. False if fullscreen was exited.</returns>
        public static bool ToggleWindowUnderCursorFullscreen()
        {
            var mouseOverWin = EditorFullscreenState.GetMouseOverWindow();
            if (mouseOverWin != null)
            {
                return EditorFullscreenState.ToggleFullscreenUsingOptions(mouseOverWin, mouseOverWin.GetType(), EditorFullscreenSettings.settings.windowUnderCursor, false, true);
            }
            else return false;
        }

        /// <summary>
        /// Toggles the top toolbar for the currently focused fullscreen window. (Only applies to Scene View, Game View, and Main Window, which have top toolbars).
        /// </summary>
        public static void ToggleTopToolbar()
        {
            if (EditorFullscreenState.RunAfterInitialStateLoaded(ToggleTopToolbar)) return;
            EditorFullscreenState.ToggleToolbarInFullscreen();
        }

        /// <summary>
        /// Closes all fullscreen editor windows.
        /// </summary>
        /// <returns>True if at least one fullscreen window was closed.</returns>
        public static bool CloseAllEditorFullscreenWindows()
        {
            bool closedAtLeastOneFullscreen = false;
            int numOfClosedFullscreens = 0;
            EWFDebugging.LogLine("Closing all fullscreen windows.");
            try
            {
                var allWinStates = EditorFullscreenState.fullscreenState.window.ToArray();
                foreach (var win in allWinStates)
                {
                    if (win.EditorWin != null && win.WindowType != EditorFullscreenState.MainWindowType)
                    {
                        if (win.IsFullscreen)
                        {
                            closedAtLeastOneFullscreen = true;
                            if (EditorDisplay.ClosestToPoint(win.FullscreenAtPosition).Locked) { EditorFullscreenState.RunAfterDisplayNotLocked(win.FullscreenAtPosition, () => CloseAllEditorFullscreenWindows()); return true; }
                            if (settings.debugModeEnabled) EWFDebugging.Log("Closing fullscreen for window, title: " + win.WindowTitle + " type: " + win.WindowType + " FullscreenAtPosition: " + win.FullscreenAtPosition + " Fullscreen in Bounds: " + win.ScreenBounds);
                            win.EditorWin.SetFullscreen(false);
                            win.containerWindow = null;
                            win.EditorWin = null; //Causes the fullscreen state to be removed in CleanDeletedWindows();
                            numOfClosedFullscreens++;
                        }
                    }
                }
            }
            catch (Exception e)
            {
                EWFDebugging.LogError("Error when closing all fullscreen windows: " + e.Message);
            }

            if (EditorMainWindow.IsFullscreen())
            {
                closedAtLeastOneFullscreen = true;
                numOfClosedFullscreens++;
                EWFDebugging.Log("Closing main window fullscreen.");
            }
            EditorMainWindow.SetFullscreen(false);

            EditorFullscreenState.fullscreenState.CleanDeletedWindows();
            EditorFullscreenState.TriggerFullscreenEvent(null, null, Vector2.zero, closedAtLeastOneFullscreen);
            EWFDebugging.LogLine("numOfClosedFullscreens: " + numOfClosedFullscreens);
            return closedAtLeastOneFullscreen;
        }

        public static bool ResetToDefaultLayout()
        {
            bool closedAtLeastOneFullscreen = CloseAllEditorFullscreenWindows();
            WindowLayoutUtility.ReloadDefaultWindowPrefs();
            return closedAtLeastOneFullscreen;
        }

        /**********************************************/
        /************ Other Public Methods ************/
        /**********************************************/

        /// <summary>
        /// Returns true if a window type is fullscreen on any screen.
        /// </summary>
        public static bool WindowTypeIsFullscreen(Type windowType)
        {
            return WindowTypeIsFullscreen(windowType, null);
        }

        /// <summary>
        /// Returns true if a window type is fullscreen on any screen, excluding the given fullscreen state.
        /// </summary>
        private static bool WindowTypeIsFullscreen(Type windowType, EditorFullscreenState.WindowFullscreenState excludeState)
        {
            bool foundOneFullscreen = false;
            foreach (var state in EditorFullscreenState.fullscreenState.window)
            {
                if (state != excludeState && state.WindowType == windowType && state.IsFullscreen)
                {
                    foundOneFullscreen = true;
                    break;
                }
            }
            return foundOneFullscreen;
        }

        /// <summary>
        /// Returns true if an editor window type is fullscreen on the screen at the specified position.
        /// </summary>
        public static bool WindowTypeIsFullscreenOnScreenAtPosition(Type windowType, Vector2 atPosition)
        {
            return EditorFullscreenState.WindowTypeIsFullscreenOnScreenAtPosition(windowType, atPosition);
        }

        /// <summary>
        /// Toggle fullscreen at the current mouse position for the window with the specified type. Shows the toolbar if applicable.
        /// </summary>
        public static bool ToggleFullscreenAtMousePosition(Type windowType)
        {
            return EditorFullscreenState.ToggleFullscreenAtMousePosition(windowType, true);
        }

        /// <summary>
        /// Toggle fullscreen at the current mouse position for the window with the specified type.
        /// </summary>
        public static bool ToggleFullscreenAtMousePosition(Type windowType, bool showTopToolbar)
        {
            return EditorFullscreenState.ToggleFullscreenAtMousePosition(windowType, showTopToolbar);
        }
        /// <summary>
        /// Toggle fullscreen for a window type (Creates a new fullscreen window if none already exists).
        /// Enters fullscreen on the primary screen.
        /// </summary>
        /// <param name="windowType">The type of the window to create a fullscreen for.</param>
        /// <returns>True if the window type became fullscreen. False if fullscreen was exited.</returns>
        public static bool ToggleFullscreen(Type windowType)
        {
            return EditorFullscreenState.ToggleFullscreen(windowType);
        }

        /// <summary>
        /// Toggle fullscreen for a window type, on the screen at a position. Shows the toolbar if applicable.
        /// </summary>
        /// <param name="windowType">The type of the window to create a fullscreen for.</param>
        /// <param name="atPosition">Fullscreen will be toggled on the screen which is at this position.</param>
        /// <returns>True if the window type became fullscreen. False if fullscreen was exited.</returns>
        public static bool ToggleFullscreen(Type windowType, Vector2 atPosition)
        {
            return EditorFullscreenState.ToggleFullscreen(windowType, atPosition, true, false);
        }

        /// <summary>
        /// Toggle fullscreen for a window type, on the screen at a position.
        /// </summary>
        /// <param name="windowType">The type of the window to create a fullscreen for.</param>
        /// <param name="atPosition">Fullscreen will be toggled on the screen which is at this position.</param>
        /// <param name="showTopToolbar">Show the top toolbar by default if opening a fullscreen.</param>
        /// <returns>True if the window type became fullscreen. False if fullscreen was exited.</returns>
        public static bool ToggleFullscreen(Type windowType, Vector2 atPosition, bool showTopToolbar)
        {
            return EditorFullscreenState.ToggleFullscreen(windowType, atPosition, showTopToolbar, false);
        }

        /// <summary>
        /// Toggle fullscreen for the game view, on the screen at a position.
        /// </summary>
        /// <param name="atPosition">Fullscreen will be toggled on the screen which is at this position.</param>
        /// <param name="showTopToolbar">Show the top toolbar by default if opening a fullscreen.</param>
        /// <returns>True if the game view became fullscreen. False if fullscreen was exited.</returns>
        public static bool ToggleGameViewFullscreen(Vector2 atPosition, bool showTopToolbar)
        {
            return ToggleFullscreen(EditorFullscreenState.GameViewType, atPosition, showTopToolbar);
        }

        /// <summary>
        /// Exit all game views except for a single one (Multiple game views showing at the same time can cause FPS drops in-game)
        /// </summary>
        /// <param name="exceptForThisWindow">This is the only exception. Will not be closed.</param>
        public static void ExitAllGameViews(EditorWindow exceptForThisWindow)
        {
            var gameViews = (EditorWindow[])Resources.FindObjectsOfTypeAll(EditorFullscreenState.GameViewType);
            var exceptState = EditorFullscreenState.FindWindowState(exceptForThisWindow);
            foreach (var win in gameViews)
            {
                var state = EditorFullscreenState.FindWindowState(win);
                if (exceptState != null && exceptState.CreatedAtGameStart && state != null && state.CreatedAtGameStart)
                {
                    //These game windows were created together at game start, so don't close them.
                    continue;
                }
                else
                {
                    if (win != exceptForThisWindow)
                    {
                        state.CloseOnExitFullscreen = true;
                        win.SetFullscreen(false);
                    }
                }
            }
        }

        /// <summary>
        /// Exits the fullscreen game views.
        /// </summary>
        /// <param name="onlyThoseCreatedAtGameStart">If true, only exits the game views which were created when the game was started.</param>
        public static void ExitGameFullscreens(bool onlyThoseCreatedAtGameStart)
        {
            EditorFullscreenState.RunOnLoad methodToRun = ExitGameFullscreensAll;
            if (onlyThoseCreatedAtGameStart) methodToRun = ExitGameFullscreensOnlyThoseCreatedAtGameStart;
            if (EditorFullscreenState.RunAfterInitialStateLoaded(methodToRun)) return;

            var fullscreenGameWindows = new List<EditorWindow>();
            foreach (var state in EditorFullscreenState.fullscreenState.window)
            {
                if (state.EditorWin != null && state.WindowType == EditorFullscreenState.GameViewType && state.EditorWin.IsFullscreen())
                {
                    if (!onlyThoseCreatedAtGameStart || state.CreatedAtGameStart)
                        fullscreenGameWindows.Add(state.EditorWin);
                }
            }
            foreach (var gameWin in fullscreenGameWindows)
            {
                gameWin.SetFullscreen(false);
            }
        }
        private static void ExitGameFullscreensAll()
        {
            ExitGameFullscreens(false);
        }
        private static void ExitGameFullscreensOnlyThoseCreatedAtGameStart()
        {
            ExitGameFullscreens(true);
        }

        /// <summary>
        /// Triggers a Fullscreen Hotkey.
        /// </summary>
        /// <param name="keyCode">The key code of the hotkey to be triggered.</param>
        /// <param name="modifiers">The modifiers of the hotkey to be triggered.</param>
        /// <returns></returns>
        internal static bool TriggerFullscreenHotkey(KeyCode keyCode, EventModifiers modifiers)
        {
            if (EditorInput.performedHotkeyActionThisUpdate) return false; //Already triggered the hotkey

            EWFDebugging.Begin();
            bool setFullscreen = false;
            bool fullscreenHotkeyTriggered = true;
            var settings = EditorFullscreenSettings.settings;
            if (settings.debugModeEnabled) EWFDebugging.LogLine("Triggered hotkey: " + EditorInput.GetKeysDownString(keyCode, modifiers) + " (key " + keyCode.ToKeyString() + " modifiers " + modifiers.ToString() + ")");
            EditorDisplay.ClearCachedDisplays();

            EWFDebugging.StartTimer("Check hotkey and fullscreen");
            if (CheckHotkeyTriggered(keyCode, modifiers, settings.closeAllFullscreenWindows)) setFullscreen = CloseAllEditorFullscreenWindows(); //In this case setFullscreen is set to true if at least one fullscreen was closed.
            else if (CheckHotkeyTriggered(keyCode, modifiers, settings.resetToDefaultLayout)) setFullscreen = ResetToDefaultLayout();
            else if (CheckHotkeyTriggered(keyCode, modifiers, settings.mainUnityWindow)) setFullscreen = ToggleMainWindowFullscreen();
            else if (CheckHotkeyTriggered(keyCode, modifiers, settings.sceneWindow)) setFullscreen = ToggleSceneViewFullscreen();
            else if (CheckHotkeyTriggered(keyCode, modifiers, settings.gameWindow)) setFullscreen = ToggleGameViewFullscreen(false, settings.gameWindow.OptionID);
            else if (CheckHotkeyTriggered(keyCode, modifiers, settings.currentlyFocusedWindow)) setFullscreen = ToggleFocusedWindowFullscreen();
            else if (CheckHotkeyTriggered(keyCode, modifiers, settings.windowUnderCursor)) setFullscreen = ToggleWindowUnderCursorFullscreen();
            else if (CheckHotkeyTriggered(keyCode, modifiers, settings.toggleTopToolbar)) ToggleTopToolbar();
            else
            {
                fullscreenHotkeyTriggered = false;

                //Check if a custom window hotkey is triggered
                if (settings.customWindows != null)
                {
                    for (int i = 0; i < settings.customWindows.Count; i++)
                    {
                        if (CheckHotkeyTriggered(keyCode, modifiers, settings.customWindows[i]))
                        {
                            if (settings.customWindows[i].isGameView)
                                setFullscreen = ToggleGameViewFullscreen(false, settings.customWindows[i].OptionID);
                            else
                                setFullscreen = EditorFullscreenState.ToggleFullscreenUsingOptions(settings.customWindows[i].WindowType, settings.customWindows[i]);
                            fullscreenHotkeyTriggered = true;
                            break;
                        }
                    }
                }
            }

            EWFDebugging.LogTime("Check hotkey and fullscreen");

            if (fullscreenHotkeyTriggered)
                triggeredHotkey = null; //Reset the triggered hotkey after fullscreen is toggled.

            if (FullscreenHotkeyEventHandler != null && fullscreenHotkeyTriggered)
            {
                FullscreenHotkeyEventHandler.Invoke(keyCode, modifiers, setFullscreen);
            }
            EWFDebugging.LogLine("fullscreenHotkeyTriggered: " + fullscreenHotkeyTriggered + ", setFullscreen: " + setFullscreen);
            if (fullscreenHotkeyTriggered) {
                EWFDebugging.PrintLog();
                EditorInput.performedHotkeyActionThisUpdate = true;
            }
            return fullscreenHotkeyTriggered;
        }

        /// <summary>
        /// Triggers a Fullscreen Hotkey
        /// </summary>
        /// <param name="fullscreenOption">The fullscreen option to trigger the hotkey of.</param>
        internal static void TriggerFullscreenHotkey(EditorFullscreenSettings.FullscreenOption fullscreenOption)
        {
            TriggerFullscreenHotkey(fullscreenOption.hotkey, fullscreenOption.modifiers);
        }

        /**********************************************/
        /************ Private Methods *****************/
        /**********************************************/
        [InitializeOnLoadMethod]
        private static void Initialize()
        {
            FullscreenHotkeyEventHandler = null;

#if UNITY_2017_2_OR_NEWER
            EditorApplication.playModeStateChanged += PlayModeStateChanged;
#else
            EditorApplication.playmodeStateChanged += PlayModeStateChanged;
#endif

            EditorInput.KeyEventHandler -= EventController;
            EditorInput.KeyEventHandler += EventController;
        }

        private static void EventController()
        {
            if (Event.current.type == EventType.KeyDown
                && Event.current.commandName != "HotkeyConflictResendEvent")
            {
                var keyCode = Event.current.keyCode;
                var modifiers = Event.current.modifiers;
                var hotkeyWasTriggered = TriggerFullscreenHotkey(keyCode, modifiers);

                if (hotkeyWasTriggered) {
                    Event.current.Use();
                }
            }
        }

        private static bool CheckHotkeyTriggered(KeyCode keyCode, EventModifiers modifiers, EditorFullscreenSettings.FullscreenOption fullscreenOption)
        {
            if (keyCode == KeyCode.None) return false;
            if (EditorInput.HotkeysMatch(keyCode, modifiers, fullscreenOption.hotkey, fullscreenOption.modifiers))
            {
                if (EditorFullscreenSettingsWindow.window != null && EditorFullscreenSettingsWindow.window.IsFocusedOnHotkeyField())
                {
                    //Don't trigger the fullscreen hotkey if currently focused on the hotkey field in the settings window.
                    EditorFullscreenSettingsWindow.window.HotkeyConflictResendEvent(keyCode, modifiers); //Resend the hotkey event to the settings window so the hotkey can be changed.
                    return false;
                }
                else
                {
                    //Trigger the fullscreen hotkey
                    triggeredHotkey = fullscreenOption;
                    return true;
                }
            }
            return false;
        }

#if UNITY_2017_2_OR_NEWER
        private static void PlayModeStateChanged(PlayModeStateChange state)
        {
            PlayModeStateChanged();
        }
#endif

        private static void PlayModeStateChanged()
        {
            bool startingPlay = !EditorApplication.isPlaying && EditorApplication.isPlayingOrWillChangePlaymode;
            bool stoppedPlay = !EditorApplication.isPlaying && !EditorApplication.isPlayingOrWillChangePlaymode;

            if (startingPlay)
            {
                if (settings.AllGameWindows.Find(w => w.openOnGameStart) != null)
                    ToggleGameViewFullscreen(true, 0);
            }
            else if (stoppedPlay)
            {
                if (settings.closeFullscreenOnGameStop == EditorFullscreenSettings.CloseFullscreenOnGameStop.AllFullscreenGameWindows)
                {
                    ExitGameFullscreens(false);
                }
                else if (settings.closeFullscreenOnGameStop == EditorFullscreenSettings.CloseFullscreenOnGameStop.FullscreensCreatedAtGameStart)
                {
                    ExitGameFullscreens(true);
                }
            }
        }
    }
}