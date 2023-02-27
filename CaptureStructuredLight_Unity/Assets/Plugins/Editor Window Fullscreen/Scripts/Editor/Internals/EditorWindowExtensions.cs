/* 
 * Author:  Johanan Round
 * Package: Editor Window Fullscreen
 * License: Unity Asset Store EULA (Editor extension asset. Requires 1 license per machine.)
 */

using UnityEngine;
using UnityEditor;
using FS = EditorWindowFullscreen.EditorFullscreenState;

using System;
using System.IO;
using System.Reflection;
using System.Collections.Generic;
using System.Runtime.InteropServices;


namespace EditorWindowFullscreen
{
    public static class EditorWindowExtensions
    {

        public enum ShowMode
        {
            NormalWindow,
            PopupMenu,
            Utility,
            NoShadow,
            MainWindow,
            AuxWindow
        }

        public static EditorWindow CreateWindow(Type windowType)
        {
            EditorWindow newEditorWin = (EditorWindow)ScriptableObject.CreateInstance(windowType);
            return newEditorWin;
        }

        /// <summary> Show the EditorWindow with a specified mode </summary>
        public static void ShowWithMode(this EditorWindow editorWindow, ShowMode showMode)
        {
            MethodInfo showWithMode = typeof(EditorWindow).GetMethod("ShowWithMode", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            showWithMode.Invoke(editorWindow, new object[] { (int)showMode });
        }

        /// <summary> Make the EditorWindow fullscreen. </summary>
        public static void SetFullscreen(this EditorWindow editorWindow)
        {
            editorWindow.SetFullscreen(true);
        }

        /// <summary> Make the EditorWindow fullscreen, or return to how it was. </summary>
        public static void SetFullscreen(this EditorWindow editorWindow, bool setFullscreen)
        {
            editorWindow.SetFullscreen(setFullscreen, editorWindow.GetPointOnWindow());
        }

        /// <summary> Make the EditorWindow fullscreen, or return to how it was. Opens the fullscreen window on the screen at a specified position. </summary>
        public static void SetFullscreen(this EditorWindow editorWindow, bool setFullscreen, Vector2 atPosition)
        {
            if (editorWindow == null) return; //It's possible it's been exited while waiting for the display to unlock.
            Type windowType = editorWindow.GetWindowType();

#if UNITY_EDITOR_OSX
            var fullscreenOnDisplay = EditorDisplay.ClosestToPoint(atPosition);

            if (fullscreenOnDisplay.Locked)
            {
                FS.RunAfterDisplayNotLocked(atPosition, () => SetFullscreen(editorWindow, setFullscreen, atPosition));
                return;
            }
#endif

            var fullscreenState = EditorFullscreenState.FindWindowState(editorWindow);
            CursorLockMode currentCursorLockMode = Cursor.lockState;

            if (setFullscreen == false)
            {
                if (fullscreenState.EditorWin != null)
                {
                    if (fullscreenState.CloseOnExitFullscreen)
                    {
#if UNITY_EDITOR_OSX
                        editorWindow.SetIdentifierTitle();

                        if (SystemDisplay.WindowIsFullscreenWithID(fullscreenState.identifierID))
                        {
                            fullscreenOnDisplay.Locked = true;
                            SystemDisplay.RestoreWindowStyle(editorWindow, fullscreenState.PreFullscreenPosition);
                            EditorApplication.CallbackFunction exitedFullscreen = null;

                            int setCount = 0;
                            if (fullscreenState.CloseOnExitFullscreen)
                            {
                                exitedFullscreen = () =>
                                {
                                    setCount++;
                                    if (setCount > 100)
                                    {
                                        fullscreenOnDisplay.Locked = false;
                                        EditorApplication.update -= exitedFullscreen;
                                        if (fullscreenState.ShowTopTabs)
                                        {
                                            SystemDisplay.CloseWindow(fullscreenState.identifierID);
                                        }
                                        else
                                        {
                                            try { editorWindow.Close(); } catch (Exception e) { EWFDebugging.LogError("Error closing window:" + e.ToString()); }
                                        }
                                        FS.SaveFullscreenState();
                                        if (EditorMainWindow.IsFullscreenAtPosition(atPosition)) SystemDisplay.FocusMainWindow(); //Return to main window space if we're on the same screen.
                                        FS.TriggerFullscreenEvent(editorWindow, windowType, atPosition, setFullscreen);
                                    }
                                    else if (setCount == 100 || !SystemDisplay.WindowIsFullscreenWithID(fullscreenState.identifierID))
                                    {
                                        setCount = 1000;
                                    }
                                };
                                EditorApplication.update += exitedFullscreen;
                            }
                        }
                        else
                        {
                            if (fullscreenState.CloseOnExitFullscreen)
                            {
                                if (fullscreenState.ShowTopTabs)
                                {
                                    SystemDisplay.CloseWindow(fullscreenState.identifierID);
                                }
                                else
                                {
                                    try { editorWindow.Close(); } catch (Exception e) { EWFDebugging.LogError("Error closing window:" + e.ToString()); }
                                }
                            }
                        }
#else
                        //Close the window
                        if (fullscreenState.ShowTopTabs)
                        {
                            var containerWin = editorWindow.GetContainerWindow();
                            if (EditorMainWindow.containerClose != null)
                            {
                                EditorMainWindow.containerClose.Invoke(containerWin, null);
                            }
                            else editorWindow.Close();
                        }
                        else
                        {
                            editorWindow.Close();
                        }
#endif
                    }
                    else
                    {

#if UNITY_EDITOR_OSX
                        editorWindow.RemoveNotification();
                        editorWindow.SetIdentifierTitle();

                        fullscreenOnDisplay.Locked = true;
                        if (SystemDisplay.WindowIsFullscreenWithID(fullscreenState.identifierID))
                        {
                            var mousePos = EditorInput.MousePosition;
                            SystemDisplay.RestoreWindowStyle(editorWindow, fullscreenState.PreFullscreenPosition); //Restore window style

                            //Update position even though haven't finished exiting fullscreen, to smooth animation.
                            if (!fullscreenState.ScreenBounds.Contains(new Vector2(fullscreenState.PreFullscreenPosition.center.x, fullscreenState.PreFullscreenPosition.yMin)))
                            {
                                var intermediatePos = fullscreenState.PreFullscreenPosition.CenterRectInBounds(fullscreenState.ScreenBounds);
                                EditorInput.SystemInput.SetMousePosition((int)intermediatePos.x, (int)intermediatePos.y, false);
                                editorWindow.SetContainerPosition(intermediatePos);
                                editorWindow.TriggerOnResizedAll();
                            }
                            else
                            {
                                EditorInput.SystemInput.SetMousePosition((int)fullscreenState.PreFullscreenPosition.x, (int)fullscreenState.PreFullscreenPosition.y, false);
                                editorWindow.SetContainerPosition(fullscreenState.PreFullscreenPosition);
                                editorWindow.TriggerOnResizedAll();
                            }
                            EditorInput.SystemInput.SetMousePosition((int)mousePos.x, (int)mousePos.y, false);
                        }


                        EditorApplication.CallbackFunction exitedFullscreen = null;
                        int setCount = 0;
                        exitedFullscreen = () =>
                        {
                            setCount++;
                            if (setCount >= 101)
                            {
                                EditorApplication.update -= exitedFullscreen;
                                fullscreenOnDisplay.Locked = false;
                            }
                            else if (setCount == 100 || !SystemDisplay.WindowIsFullscreenWithID(fullscreenState.identifierID))
                            {
                                setCount = 100;
                                var mousePos = EditorInput.MousePosition;
                                var screenCentre = EditorDisplay.ClosestToPoint(fullscreenState.PreFullscreenPosition.center).Bounds.center;
                                EditorInput.SystemInput.SetMousePosition((int)screenCentre.x, (int)screenCentre.y, false);
#else
                        //Restore window style
                        SystemDisplay.RestoreWindowStyle(editorWindow, fullscreenState.PreFullscreenPosition);
#endif

                        //Restore the window
                        editorWindow.minSize = fullscreenState.PreFullscreenMinSize;
                        editorWindow.maxSize = fullscreenState.PreFullscreenMaxSize;
                        if (fullscreenState.ShowTopTabs)
                        {
                            editorWindow.SetContainerPosition(fullscreenState.PreFullscreenPosition);
                        }
                        else
                        {
                            editorWindow.position = fullscreenState.PreFullscreenPosition;
                        }


                        if (editorWindow.maximized != fullscreenState.PreFullscreenMaximized)
                            editorWindow.maximized = fullscreenState.PreFullscreenMaximized;

                        //Restore window title
                        if (!fullscreenState.ShowTopTabs) editorWindow.SetWindowTitle(fullscreenState.WindowTitle);

#if UNITY_EDITOR_OSX
                                if (EditorMainWindow.IsFullscreenAtPosition(atPosition)) SystemDisplay.FocusMainWindow(); //Return to main window space if we're on the same screen.
                                editorWindow.TriggerOnResizedAll();
                                editorWindow.Repaint();
                                editorWindow.Focus();
                                EditorInput.SystemInput.SetMousePosition((int)mousePos.x, (int)mousePos.y, false);
                                FS.SaveFullscreenState();
                                FS.TriggerFullscreenEvent(editorWindow, windowType, atPosition, setFullscreen);

                            }
                        };
                        EditorApplication.update += exitedFullscreen;
#endif

                        //Schedule fullscreen state for deletion
                        fullscreenState.EditorWin = null;

                    }
                }

                if (editorWindow.GetWindowType() == FS.GameViewType)
                {
                    //Unlock the cursor when exiting game fullscreen
#if UNITY_2018_1_OR_NEWER
                    FS.GameViewType.GetMethod("OnLostFocus", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance).Invoke(editorWindow, null);
#else
                    Unsupported.SetAllowCursorLock(false);
#endif
                }

                if (fullscreenState.UnfocusedGameViewOnEnteringFullscreen == true)
                {
                    //Refocus the first docked game view
                    var gameView = GetDockedGameView(editorWindow, false);
                    if (gameView != null) gameView.Focus();
                }

            }
            else
            {
#if UNITY_EDITOR_OSX
                bool exitedFullscreens = editorWindow.ExitFullscreenForOtherWindowsOnScreen(atPosition);
                if (exitedFullscreens)
                {
                    //Wait for the other fullscreens to finish exiting
                    FS.RunAfterDisplayNotLocked(atPosition, () => SetFullscreen(editorWindow, setFullscreen, atPosition));
                    return;
                }
#else
                editorWindow.ExitFullscreenForOtherWindowsOnScreen(atPosition);
#endif

                if (!fullscreenState.IsFullscreen)
                {
                    if (fullscreenState.ShowTopTabs)
                    {
                        if (editorWindow.IsInMainWin())
                            fullscreenState.PreFullscreenPosition = editorWindow.position;
                        else
                            fullscreenState.PreFullscreenPosition = editorWindow.GetContainerPosition();
                    }
                    else
                    {
                        fullscreenState.PreFullscreenPosition = editorWindow.position;
                        fullscreenState.PreFullscreenPosition.y -= FS.windowTopPadding;
                    }

                    fullscreenState.PreFullscreenMinSize = editorWindow.minSize;
                    fullscreenState.PreFullscreenMaxSize = editorWindow.maxSize;
                    fullscreenState.PreFullscreenMaximized = editorWindow.maximized;
                }

                if (!editorWindow.IsFullscreen())
                {
                    editorWindow.maximized = false;

                    if (fullscreenState.ShowTopTabs)
                    {
                        if (fullscreenState.CreatedNewWindow && !fullscreenState.fullscreenPositionWasSet) //Only show once
                            editorWindow.Show();
                    }
                    else
                    {
                        FS.calculatedBorderlessOffsets = false;
#if UNITY_EDITOR_WIN
                        editorWindow.ShowWithMode(ShowMode.PopupMenu);
                        editorWindow.SetBorderlessPosition(new Rect(atPosition.x, atPosition.y, 100, 100));
#else
                        if (fullscreenState.CreatedNewWindow && !fullscreenState.fullscreenPositionWasSet)
                        { //Only show once
#if UNITY_2019_1_OR_NEWER
                            editorWindow.ShowWithMode(ShowMode.Utility);
#else
                            editorWindow.ShowWithMode(ShowMode.NormalWindow); //Issue R11
                            editorWindow.SetShowMode(ShowMode.PopupMenu); //Fixes clicking on buttons on the toolbar in <U2018
#endif
                        }
#endif
                    }

                    fullscreenState.FullscreenAtPosition = atPosition;

                }
                else if (fullscreenState.IsFullscreen)
                {
#if UNITY_EDITOR_WIN
                    //If already fullscreen, resize slightly to make sure the taskbar gets covered (E.g. when loading fullscreen state on startup)
                    if (fullscreenState.ShowTopTabs)
                    {
                        var tempBounds = editorWindow.GetContainerPosition();
                        tempBounds.yMax -= 1;
                        editorWindow.SetContainerPosition(tempBounds);
                    }
                    else
                    {
                        var tempBounds = editorWindow.position;
                        tempBounds.yMax -= 1;
                        editorWindow.SetBorderlessPosition(tempBounds);
                    }
#endif
                }

                editorWindow.SetIdentifierTitle();

                fullscreenState.ScreenBounds = editorWindow.MakeFullscreenWindow(!fullscreenState.ShowTopToolbar, atPosition);

                fullscreenState.WindowName = editorWindow.name;
                fullscreenState.EditorWin = editorWindow;
                fullscreenState.IsFullscreen = true;
                FS.lastWinStateToChangeFullscreenStatus = fullscreenState;

                try
                {
                    if (editorWindow.GetWindowType() == FS.GameViewType)
                    {
                        if (EditorFullscreenSettings.settings.improveFpsOptions == EditorFullscreenSettings.ImproveFPSOptions.CloseAllOtherGameWindows)
                        {
                            EditorFullscreenController.ExitAllGameViews(editorWindow);
                        }
                        else
                        {
                            //Usability improvement for Unity bug where only one visible game window accepts input. (Unfocus docked game views if opening fullscreen view on the same screen.)
                            //Also unfocus the docked game view to avoid FPS drops if that option is enabled.
                            var gameView = GetDockedGameView(editorWindow, true);
                            if (gameView != null)
                            {
                                bool onSameDisplay = EditorDisplay.ClosestToPoint(gameView.position.center).Bounds.Contains(atPosition);
                                bool hideDockedGameView = onSameDisplay || EditorFullscreenSettings.settings.improveFpsOptions == EditorFullscreenSettings.ImproveFPSOptions.HideDockedGameView;

                                var dockArea = gameView.GetDockArea();
                                if (hideDockedGameView && dockArea != null && FS.DockAreaType != null)
                                {
                                    FieldInfo m_Panes = FS.DockAreaType.GetField("m_Panes", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                                    var dockAreaPanes = (List<EditorWindow>)m_Panes.GetValue(dockArea);

                                    foreach (var sibling in dockAreaPanes)
                                    {
                                        if (sibling.GetType() != FS.GameViewType)
                                        {
                                            sibling.Focus(); //Focus the first non-game sibling of the docked game view
                                            fullscreenState.UnfocusedGameViewOnEnteringFullscreen = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                catch (System.Exception e)
                {
                    if (EWFDebugging.Enabled)
                    {
                        Debug.LogException(e);
                        EWFDebugging.LogError(e.Message);
                    }
                }

                editorWindow.Focus();
                editorWindow.SetSaveToLayout(true); //Must save to layout or this window won't reopen on restart

                Cursor.lockState = currentCursorLockMode; //Ensure that the cursor lock mode remains the same when entering fullscreen

                //Fullscreen Notification
                if (EditorFullscreenSettings.settings.fullscreenNotification == true)
                {
                    string notificationMessage = "";
                    if (EditorFullscreenController.triggeredHotkey != null && EditorFullscreenController.triggeredHotkey.hotkey != KeyCode.None)
                    {
                        bool toggledToolbar = EditorFullscreenController.triggeredHotkey.Equals(EditorFullscreenSettings.settings.toggleTopToolbar);
                        bool inclBrackets = EditorFullscreenController.triggeredHotkey.modifiers == EventModifiers.None && (EditorFullscreenController.triggeredHotkey.hotkey == KeyCode.F || EditorFullscreenController.triggeredHotkey.hotkey.ToString().StartsWith("F") == false);
                        notificationMessage = "\u00A0\u00A0\u00A0" + (inclBrackets ? "[" : "") + EditorInput.GetKeysDownString(EditorFullscreenController.triggeredHotkey.hotkey, EditorFullscreenController.triggeredHotkey.modifiers)
                                            + (inclBrackets ? "]" : "") + "\u00A0toggled\u00A0" + (toggledToolbar ? "toolbar" : "fullscreen") + "\u00A0\u00A0\u00A0";
                    }
                    else
                    {
                        var closeAllKey = EditorFullscreenSettings.settings.closeAllFullscreenWindows;
                        notificationMessage = EditorInput.GetKeysDownString(closeAllKey.hotkey, closeAllKey.modifiers) + " to exit all fullscreens";
                    }
                    var notificationContent = new GUIContent(notificationMessage);
                    editorWindow.ShowNotification(notificationContent);
                    float timeBeforeFade = Mathf.Min(10, Mathf.Max(0, EditorFullscreenSettings.settings.fullscreenNotificationDuration));
                    FieldInfo fadeOutTimeFI = editorWindow.GetType().GetField("m_FadeoutTime", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                    if (fadeOutTimeFI != null) fadeOutTimeFI.SetValue(editorWindow, (float)EditorApplication.timeSinceStartup + timeBeforeFade);
                }

                SetGameViewOptions(editorWindow, atPosition);
            }

#if UNITY_EDITOR_OSX == false
            fullscreenState.currentlyRestoringFromState = false;
#endif
            FS.SaveFullscreenState();
            FS.TriggerFullscreenEvent(editorWindow, windowType, atPosition, setFullscreen);
        }

        /// <summary> Set fullscreen with the option to show or hide the top tabs </summary>
        public static void SetFullscreen(this EditorWindow editorWindow, bool setFullscreen, bool showTopToolbar)
        {
            editorWindow.SetFullscreen(setFullscreen, editorWindow.GetPointOnWindow(), showTopToolbar);
        }

        /// <summary> Set fullscreen with the option to show or hide the top tabs </summary>
        public static void SetFullscreen(this EditorWindow editorWindow, bool setFullscreen, Vector2 atPosition, bool showTopToolbar)
        {
            var fullscreenState = EditorFullscreenState.FindWindowState(editorWindow);

            if (editorWindow.GetWindowType() == FS.GameViewType)
            {
                if (showTopToolbar && !fullscreenState.ShowTopToolbar)
                {
                    fullscreenState.CursorLockModePreShowTopToolbar = Cursor.lockState;
                    Cursor.lockState = CursorLockMode.None; //Enable cursor when top tab is enabled
                }
                else if (!showTopToolbar && fullscreenState.ShowTopToolbar)
                {
                    Cursor.lockState = fullscreenState.CursorLockModePreShowTopToolbar; //Reset cursor lock mode when top tab is disabled
                }
            }

            fullscreenState.ShowTopToolbar = showTopToolbar;
            SetFullscreen(fullscreenState.EditorWin, setFullscreen, atPosition);
        }

        /// <summary> Toggle fullscreen for a window. </summary>
        public static bool ToggleFullscreen(this EditorWindow editorWindow)
        {
            return ToggleFullscreen(editorWindow, editorWindow.GetPointOnWindow());
        }

        /// <summary> Toggle fullscreen for a window, on the screen at a specified position. </summary>
        public static bool ToggleFullscreen(this EditorWindow editorWindow, Vector2 atPosition)
        {
            if (editorWindow != null)
            {
                var fullscreenState = FS.FindWindowState(editorWindow);
                bool setFullScreen = (fullscreenState != null && !fullscreenState.fullscreenPositionWasSet) || !(editorWindow.IsFullscreenOnDisplay(EditorDisplay.ClosestToPoint(atPosition)));

                editorWindow.SetFullscreen(setFullScreen, atPosition);

                return setFullScreen;
            }
            else
            {
                SetFullscreen(null, false);
                return false;
            }
        }

        ///Get the current identifying title
        public static string GetIdentifierTitle(this EditorWindow editorWindow)
        {
            var fullscreenState = FS.FindWindowState(editorWindow);
            bool fullscreenTheContainer = fullscreenState.ShowTopTabs;
            if (fullscreenTheContainer)
            {
                return editorWindow.GetContainerTitle();
            }
            else
            {
                return editorWindow.GetWindowTitle();
            }
        }

        ///Set an identifier title on the window so that it can be found later
        public static bool SetIdentifierTitle(this EditorWindow editorWindow)
        {
            var fullscreenState = FS.FindWindowState(editorWindow);
            return SetIdentifierTitle(editorWindow, fullscreenState);
        }

        ///Set an identifier title on the window so that it can be found later
        public static bool SetIdentifierTitle(this EditorWindow editorWindow, FS.WindowFullscreenState fullscreenState)
        {
            long identifierID = fullscreenState.identifierID;
            if (identifierID == 0)
            {
                identifierID = DateTime.Now.Ticks;
            }
            string windowIdentifier = "FULLSCREEN_WINDOW_" + identifierID;
            return editorWindow.SetIdentifierTitle(fullscreenState, windowIdentifier);
        }

        ///Set an identifier title on the window so that it can be found later
        public static bool SetIdentifierTitle(this EditorWindow editorWindow, string windowIdentifier)
        {
            var fullscreenState = FS.FindWindowState(editorWindow);
            return SetIdentifierTitle(editorWindow, fullscreenState, windowIdentifier);
        }

        ///Set an identifier title on the window so that it can be found later
        public static bool SetIdentifierTitle(this EditorWindow editorWindow, FS.WindowFullscreenState fullscreenState, string windowIdentifier)
        {
            bool setSuccess;

            if (windowIdentifier == null) return false;
            if (fullscreenState.ShowTopTabs && editorWindow.IsInMainWin()) setSuccess = false; //Don't set an identifier on the main window.
            else if (fullscreenState.ShowTopTabs)
            {
                //Fullscreening the container, so set container title.
                editorWindow.SetContainerTitle(windowIdentifier); //Set the container title, which may be blank for non-popup windows.
                setSuccess = true;
            }
            else
            {
                editorWindow.SetWindowTitle(windowIdentifier, true);
                setSuccess = true;
            }
            if (setSuccess)
            {
                fullscreenState.identifierTitle = windowIdentifier;

                //Set the identifierID from the string because sometimes we only have the string.
                var split = windowIdentifier.Split('_');
                long identifierID = 0;
                if (split.Length >= 3) long.TryParse(split[2], out identifierID);
                fullscreenState.identifierID = identifierID;
            }
            return setSuccess;
        }
        ///Get the current identifying ID
        public static long GetIdentifierID(this EditorWindow editorWindow)
        {
            var fullscreenState = FS.FindWindowState(editorWindow);
            return fullscreenState.identifierID;
        }

        /// <summary> Make the EditorWindow become a fullscreen window </summary>
        private static Rect MakeFullscreenWindow(this EditorWindow editorWindow)
        {
            return editorWindow.MakeFullscreenWindow(false);
        }

        /// <summary> Make the EditorWindow into a fullscreen window, with the option to show the top tabs </summary>
        private static Rect MakeFullscreenWindow(this EditorWindow editorWindow, bool hideTopToolbar)
        {
            return editorWindow.MakeFullscreenWindow(hideTopToolbar, editorWindow.GetPointOnWindow());
        }

        /// <summary> Make the EditorWindow into a fullscreen window, with the option to show the top tabs. Opens the fullscreen window on the screen at a specified position. </summary>
        private static Rect MakeFullscreenWindow(this EditorWindow editorWindow, bool hideTopToolbar, Vector2 atPosition)
        {
            var fullscreenOnDisplay = EditorDisplay.ClosestToPoint(atPosition);
#if UNITY_2018_2_OR_NEWER || UNITY_EDITOR_OSX
            var winRect = fullscreenOnDisplay.Bounds;
#else
            var winRect = fullscreenOnDisplay.Bounds;

            if (hideTopToolbar == true)
            {
                /*Move the top tab off the screen*/
                winRect.y -= FS.topTabFullHeight;
                winRect.height += FS.topTabFullHeight;
            }
#endif

            var screenToFS = SystemDisplay.ClosestToPoint(atPosition);

#if UNITY_EDITOR_OSX
            var mousePos = EditorInput.MousePosition;
            var screenCentre = screenToFS.Bounds.center;
            EditorInput.SystemInput.SetMousePosition((int)screenCentre.x, (int)screenCentre.y, false); //Set the mouse position temporarily because popup windows always go to the mouseover screen on Mac when setting position.
#endif

            var fullscreenState = FS.FindWindowState(editorWindow);

            if (!fullscreenState.ShowTopTabs)
            {
                //Not for fullscreen-the-container windows.
                if (!fullscreenState.fullscreenPositionWasSet)
                { //Only set once when initially fullscreened, or munts toggle top toolbar.
                    editorWindow.SetBorderlessPosition(winRect, hideTopToolbar);
                    fullscreenState.fullscreenPositionWasSet = true;
                }
            }
            else if (fullscreenState.ShowTopTabs && screenToFS != null)
            {
                bool useIntermediatePos = false;
                Rect intermediatePos = Rect.zero;
                if (!winRect.Contains(new Vector2(fullscreenState.PreFullscreenPosition.center.x, fullscreenState.PreFullscreenPosition.yMin)))
                {
                    //Need to move to the other screen
                    useIntermediatePos = true;
                    intermediatePos = fullscreenState.PreFullscreenPosition.CenterRectInBounds(winRect);
                }

                //For fullscreen-the-container windows.
                if (editorWindow.IsInMainWin())
                {
                    //Get it out of the main window first
                    editorWindow.position = useIntermediatePos ? intermediatePos : fullscreenState.PreFullscreenPosition;
                }
                if (useIntermediatePos)
                {
#if UNITY_EDITOR_OSX
                    SystemDisplay.SetWindowPosition(editorWindow, (int)intermediatePos.x, (int)intermediatePos.y, (int)intermediatePos.width, (int)intermediatePos.height);
#else
                    editorWindow.SetContainerPosition(intermediatePos);
#endif
                    editorWindow.TriggerOnResizedAll();
                }

                FS.calculatedBorderlessOffsets = false;
                CalculateBorderlessOffsets(editorWindow);
                fullscreenState.fullscreenPositionWasSet = true;
            }

#if UNITY_EDITOR_OSX
            EditorInput.SystemInput.SetMousePosition((int)mousePos.x, (int)mousePos.y, false); //Restore the mouse position
            fullscreenOnDisplay.Locked = true;
            int setCount = 0;
            EditorApplication.CallbackFunction setToolbarVisibility = null;
            setToolbarVisibility = () =>
            {
                setCount++;
                if (setCount <= 1 && editorWindow != null)
                {
                    var mousePosA = EditorInput.MousePosition;
                    EditorInput.SystemInput.SetMousePosition((int)screenCentre.x, (int)screenCentre.y, false); //Set the mouse position temporarily because popup windows always go to the mouseover screen on Mac when setting position.

                    bool successA = editorWindow.SetIdentifierTitle();
                    if (!successA) EWFDebugging.LogWarning("Identifier didn't set.", true);
                    bool successB = SystemDisplay.FullscreenWindowWithID(fullscreenState.identifierID);
                    if (!successB) EWFDebugging.LogError("SystemDisplay fullscreen failed.", true);

                    editorWindow.SetContainerPosition(winRect);

                    EditorInput.SystemInput.SetMousePosition((int)mousePosA.x, (int)mousePosA.y, false); //Restore the mouse position
                }
                else if (setCount <= 100)
                {
                    if (setCount == 100 || SystemDisplay.WindowIsFullscreenWithID(fullscreenState.identifierID))
                    {
                        setCount = 1000;

                        editorWindow.TriggerOnResizedAll();

#if UNITY_2019_1_OR_NEWER == false
                        if (!fullscreenState.ShowTopTabs)
                        {
                            //2018 and earlier needs to set position after fullscreen for borderless windows.
                            var mousePosA = EditorInput.MousePosition;
                            EditorInput.SystemInput.SetMousePosition((int)screenCentre.x, (int)screenCentre.y, false);
                            editorWindow.SetBorderlessPosition(winRect, hideTopToolbar);
                            EditorInput.SystemInput.SetMousePosition((int)mousePosA.x, (int)mousePosA.y, false);
                        }
#endif
                        editorWindow.SetToolbarVisibilityAtPos(winRect, hideTopToolbar, false);

                        editorWindow.Repaint();
                        editorWindow.Focus();
                        fullscreenState.currentlyRestoringFromState = false;
                        FS.SaveFullscreenState();
                    }
                }
                else
                {
                    editorWindow.Focus(); //Re-focus on the second loop, sometimes the window loses focus after the first loop when toggling toolbar.
                    fullscreenOnDisplay.Locked = false;
                    EditorApplication.update -= setToolbarVisibility;
                }
            };
            EditorApplication.update += setToolbarVisibility;

            return winRect;

#elif UNITY_EDITOR_WIN
            var mainWindowDisp = SystemDisplay.WithMainWindow();
            if (screenToFS == null && EWFDebugging.Enabled)
            {
                string warning = "Could not find any system display. Some multi-display setups may not work if display scaling differs.";
                Debug.LogWarning(warning);
                EWFDebugging.LogWarning(warning);
            }
            if (mainWindowDisp == null && EWFDebugging.Enabled)
            {
                string warning = "Could not find the main window display. Some multi-display setups may not work if display scaling differs.";
                Debug.LogWarning(warning);
                EWFDebugging.LogWarning(warning);
            }

            //Call system SetWindowPosition to make sure the window is fullscreen and covers the taskbar
            SystemDisplay.MakeWindowCoverTaskBar(editorWindow, screenToFS);

            editorWindow.TriggerOnResizedAll();

            //Hide the top toolbar if necessary.
            editorWindow.SetToolbarVisibilityAtPos(winRect, hideTopToolbar, false);

            winRect = editorWindow.position;
            return winRect;
#elif UNITY_EDITOR_LINUX
            editorWindow.SetBorderlessPosition(winRect, hideTopToolbar);
            return editorWindow.position;
#else
            Debug.LogError("Unknown/Unsupported Operating System");
            return editorWindow.position;
#endif
        }

        /// <summary> Make the EditorWindow borderless and give it an accurate position and size </summary>
        public static void SetBorderlessPosition(this EditorWindow editorWindow, Rect position)
        {
            SetBorderlessPosition(editorWindow, position, false);
        }

        /// <summary> Make the EditorWindow borderless and give it an accurate position and size. Optionally hide the top toolbar of the window if one exists. </summary>
        public static void SetBorderlessPosition(this EditorWindow editorWindow, Rect position, bool hideTopToolbar)
        {
#if UNITY_EDITOR_OSX == false
            position = editorWindow.SetToolbarVisibilityAtPos(position, hideTopToolbar, true);
#endif

            /*Make sure the window is borderless*/
            if (editorWindow.minSize != editorWindow.maxSize)
            {
                editorWindow.position = position;
                editorWindow.minSize = position.size;
                editorWindow.maxSize = position.size;
            }
            else
            {
                editorWindow.position = position;
            }

            CalculateBorderlessOffsets(editorWindow);

            /*Re-adjust the position of the window, taking into account the Y offset and top padding so that the window ends up where it should be*/
            position.y -= FS.windowTopPadding;

            editorWindow.minSize = position.size;
            editorWindow.maxSize = position.size;
            editorWindow.position = position;

#if UNITY_EDITOR_OSX == false
            editorWindow.SetToolbarVisibilityAtPos(position, hideTopToolbar, false);
#endif
        }

        private static void CalculateBorderlessOffsets(EditorWindow editorWindow)
        {
            if (!FS.calculatedBorderlessOffsets)
            {
                object hostView = editorWindow.GetHostView();
                if (hostView == null) return;
                FS.calculatedBorderlessOffsets = true;

                /*Attempt to find the top tab full height and windowTopPadding. If they can't be found the initial values are used.*/
                try
                {
                    MethodInfo GetBorderSize = FS.HostViewType.GetMethod("GetBorderSize", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                    RectOffset hostViewBorderSize = (RectOffset)GetBorderSize.Invoke(hostView, null);
                    FS.topTabFullHeight = hostViewBorderSize.top;
#if UNITY_EDITOR_WIN
                    FS.topTabFullHeight -= 3;
#else
                    FS.topTabFullHeight -= 2;
#endif
                    FieldInfo kTabHeight = null;
                    if (FS.DockAreaType != null) kTabHeight = FS.DockAreaType.GetField("kTabHeight", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
                    float tabHeight = FS.sceneViewToolbarHeight;
                    if (kTabHeight != null) tabHeight = (float)kTabHeight.GetValue(hostView);
                    FS.windowTopPadding = Mathf.Max(0, -tabHeight);
                }
                catch (System.Exception e)
                {
                    if (EWFDebugging.Enabled)
                    {
                        Debug.LogException(e);
                        EWFDebugging.LogError(e.Message);
                    }
                }
            }
        }

        /// <summary>
        /// For internal use only. Don't use this to show/hide the toolbar. Instead use a SetFullscreen overload.
        /// </summary>
        internal static Rect SetToolbarVisibilityAtPos(this EditorWindow editorWindow, Rect position, bool hideTopToolbar, bool useFallbackHide)
        {
            //Set toolbar visibility
            Type windowType = editorWindow.GetWindowType();
            FieldInfo toolbarVisible = editorWindow.GetType().GetField("toolbarVisible");
            bool hasToolbarProperty = toolbarVisible != null;
            object hostView = editorWindow.GetHostView();
            if (hostView == null) { EWFDebugging.LogWarning("Host View does not exist", true); return position; }
            PropertyInfo viewPos = null;
            MethodInfo setPos = null;
            bool offsetHostViewToHideToolbar = false;
            bool fallbackOffsetToolbar = false;
            var fullscreenState = FS.FindWindowState(editorWindow);
            bool useDockToolbarHeight = fullscreenState != null && fullscreenState.ShowTopTabs;
            bool hasInnerToolbar = windowType == typeof(SceneView) || windowType == FS.GameViewType;
            float toolbarHeight = useDockToolbarHeight ? FS.topTabFullHeight + (hasInnerToolbar && !hasToolbarProperty ? FS.sceneViewToolbarHeight : 0) : FS.sceneViewToolbarHeight;
            if (hasToolbarProperty)
            {
                //Set visibility using the property if the EditorWindow has that option.
                toolbarVisible.SetValue(editorWindow, !hideTopToolbar);
            }

            if ((hideTopToolbar && hasInnerToolbar && !hasToolbarProperty) || useDockToolbarHeight)
            {
                try
                {
                    //Offset the hostview within its container to hide the toolbar
                    viewPos = FS.ViewType.GetProperty("position", BindingFlags.Public | BindingFlags.Instance | BindingFlags.NonPublic);
                    setPos = FS.ViewType.GetMethod("SetPosition", BindingFlags.Public | BindingFlags.Instance | BindingFlags.NonPublic);
                    if (hostView == null || viewPos == null || setPos == null)
                    {
                        if (EWFDebugging.Enabled)
                        {
                            string err = "One or more properties do not exist which are required to hide the toolbar. Using fallback. hostView: " + (hostView == null ? "null" : hostView.ToString()) + " viewPos: " + (viewPos == null ? "null" : viewPos.ToString()) + " setPos: " + (setPos == null ? "null" : setPos.ToString());
                            Debug.LogError(err);
                            EWFDebugging.LogError(err);
                        }
                        fallbackOffsetToolbar = true;
                    }
                    else
                    {
                        offsetHostViewToHideToolbar = true;
                    }
                }
                catch (Exception e)
                {
                    if (EWFDebugging.Enabled) EWFDebugging.LogError("Failed to get properties. Using fallback. Full error: " + e);
                    fallbackOffsetToolbar = true;
                }
                finally
                {
                    if (useFallbackHide)
                    {
                        if (fallbackOffsetToolbar)
                        {
                            //Fallback: Offset the entire window upwards in order to hide the top toolbar on the current screen
                            position.y -= toolbarHeight;
                            position.height += toolbarHeight;
                        }
                    }
                }
            }

            if (!useFallbackHide && hideTopToolbar && offsetHostViewToHideToolbar)
            {
                //Apply the toolbar offset to the Host View, to hide the toolbar
                Rect currentViewPos;
                currentViewPos = (Rect)viewPos.GetValue(hostView, null);
                currentViewPos.y -= toolbarHeight;
                currentViewPos.height += toolbarHeight;
                setPos.Invoke(hostView, new object[] { currentViewPos });
            }
#if UNITY_EDITOR_WIN
            else if (useDockToolbarHeight && !hideTopToolbar && viewPos != null)
            {
                //var currentViewPos = (Rect)viewPos.GetValue(hostView, null);
                //currentViewPos.y += 5; //Give some padding to the top of the dock toolbar
                //currentViewPos.height -= 5;
                //setPos.Invoke(hostView, new object[] { currentViewPos });
            }
#endif
            return position;
        }

        /// <summary>
        /// Trigger the OnResize method of an Editor Window and its container.
        /// </summary>
        /// <param name="editorWindow"></param>
        private static void TriggerOnResizedAll(this EditorWindow editorWindow)
        {
            MethodInfo OnResizedC = FS.ContainerWindowType.GetMethod("OnResize", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            var containerWin = editorWindow.GetContainerWindow();
            if (OnResizedC != null && containerWin != null) OnResizedC.Invoke(containerWin, null);
            else EWFDebugging.LogWarning("ContainerWindow OnResize method doesn't exist.");
            MethodInfo OnResized = typeof(EditorWindow).GetMethod("OnResized", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (OnResized != null) OnResized.Invoke(editorWindow, null);
            else EWFDebugging.LogWarning("EditorWindow OnResized method doesn't exist.");
        }

        /// <summary>
        /// Set the game view options according to the fullscreen options for the game window.
        /// </summary>
        /// <param name="editorWindow">The game view.</param>
        private static void SetGameViewOptions(this EditorWindow editorWindow, Vector2 screenAtPosition)
        {
            if (editorWindow.GetType() != FS.GameViewType) return;

            var fullscreenState = FS.FindWindowState(editorWindow);
            PropertyInfo lowResolutionForAspectRatios = FS.GameViewType.GetProperty("lowResolutionForAspectRatios", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            //Calculate the pixels per point manually because EditorGUIUtility.pixelsPerPoint is unreliable when not called from inside the OnGUI.
            var screenPixelsWidth = EditorDisplay.ClosestToPoint(screenAtPosition).Bounds.width;
            var winPixelsWidth = editorWindow.position.width;
            var pixelsPerPoint = screenPixelsWidth / winPixelsWidth; //Relies on this method being called only after entering fullscreen

            if (fullscreenState.FullscreenOptions != null && fullscreenState.FullscreenOptions.gameViewOptions != null)
            {
                if (fullscreenState.initialOptionsWereSet) return; //Don't reset the options every time the state is loaded.
                fullscreenState.initialOptionsWereSet = true;
                var gameViewOps = fullscreenState.FullscreenOptions.gameViewOptions;

                //Game Display
                var m_TargetDisplay = FS.GameViewType.GetField("m_TargetDisplay", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
#if UNITY_2019_3_OR_NEWER
                if (m_TargetDisplay == null) m_TargetDisplay = FS.PlayModeViewType.GetField("m_TargetDisplay", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
#endif
                if (m_TargetDisplay != null) m_TargetDisplay.SetValue(editorWindow, gameViewOps.display);

                //Low Resolution Aspect Ratios
                bool setLowRes = gameViewOps.lowResolutionAspectRatios;
                if (pixelsPerPoint == 1)
                    setLowRes = false;
                if (lowResolutionForAspectRatios != null) lowResolutionForAspectRatios.SetValue(editorWindow, setLowRes, null);

                //Aspect Ratio
                var selectedSizeIndex = EditorFullscreenState.GameViewType.GetProperty("selectedSizeIndex", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (selectedSizeIndex != null) selectedSizeIndex.SetValue(editorWindow, gameViewOps.aspectRatio, null);

                //Show Stats
                var m_Stats = FS.GameViewType.GetField("m_Stats", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (m_Stats != null) m_Stats.SetValue(editorWindow, gameViewOps.stats);

                //Show Gizmos
                var m_Gizmos = FS.GameViewType.GetField("m_Gizmos", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (m_Gizmos != null) m_Gizmos.SetValue(editorWindow, gameViewOps.gizmos);

                //Values changed so apply zoom area
                var UpdateZoomAreaAndParent = FS.GameViewType.GetMethod("UpdateZoomAreaAndParent", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (UpdateZoomAreaAndParent != null) UpdateZoomAreaAndParent.Invoke(editorWindow, null);

                //Scale
                var snapZoom = FS.GameViewType.GetMethod("SnapZoom", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (snapZoom != null) snapZoom.Invoke(editorWindow, new object[] { gameViewOps.scale });

            }
            else
            {
#if UNITY_2018_2_OR_NEWER //Disable Low Resolution Aspect Ratios in the Game View if the screen scale isn't a factor of 1 (to prevent pixelization)
                if (editorWindow.GetWindowType() == FS.GameViewType && pixelsPerPoint != 1)
                {
                    if (pixelsPerPoint % 1 != 0)
                    {
                        if (lowResolutionForAspectRatios != null) lowResolutionForAspectRatios.SetValue(editorWindow, false, null);
                    }
                }
#endif
            }

        }

        /// <summary> Returns a point on the window no matter if it is currently fullscreen or non-fullscreen.</summary>
        public static Vector2 GetPointOnWindow(this EditorWindow editorWindow)
        {
            Vector2 pointOnWindow;
            var state = FS.FindWindowState(editorWindow);
            if (state != null && state.IsFullscreen)
            {
                //When the window is fullscreen sometimes position returns incorrect co-ordinates, so use the FullscreenAtPosition for fullscreen windows.
                pointOnWindow = state.FullscreenAtPosition;
            }
            else
                pointOnWindow = editorWindow.position.center;
            return pointOnWindow;
        }

        /// <summary> Returns true if the EditorWindow is currently docked in the main window </summary>
        public static bool IsInMainWin(this EditorWindow editorWindow)
        {
            return EditorMainWindow.FindContainerWindow() == editorWindow.GetContainerWindow();
        }

        /// <summary> Returns true if the EditorWindow is currently fullscreen on its current screen </summary>
        public static bool IsFullscreen(this EditorWindow editorWindow)
        {
            return IsFullscreen(editorWindow, editorWindow.GetPointOnWindow());
        }

        /// <summary> Returns true if the EditorWindow is currently fullscreen on the screen at a position </summary>
        public static bool IsFullscreen(this EditorWindow editorWindow, Vector2 atPosition)
        {
            var fullscreenState = EditorFullscreenState.FindWindowState(editorWindow);
            return fullscreenState.IsFullscreen && editorWindow.IsFullscreenOnDisplay(EditorDisplay.ClosestToPoint(atPosition));
        }

        /// <summary> Returns true if the EditorWindow is currently fullscreen on the screen at a position </summary>
        public static bool IsFullscreenOnDisplay(this EditorWindow editorWindow, EditorDisplay display)
        {
            if (display == null) return false;
            Rect containerPosition = editorWindow.GetContainerPosition(true);
            bool isFullscreenOnDisplay = false;
#if UNITY_EDITOR_WIN
            var identifierTitle = editorWindow.GetIdentifierTitle();
            if (identifierTitle == null || !identifierTitle.StartsWith("FULLSCREEN_")) return false;
            isFullscreenOnDisplay = SystemDisplay.WindowIsFullscreenOnDisplay(editorWindow, null, SystemDisplay.ContainingPoint((int)display.Bounds.center.x, (int)display.Bounds.center.y));
#else
            isFullscreenOnDisplay = containerPosition.Contains(display.Bounds.center) && display.Bounds.width == containerPosition.width || (display.Bounds.Contains(containerPosition.center) && Math.Abs(containerPosition.width - display.Bounds.width) < 1f && Math.Abs(containerPosition.height - display.Bounds.height) < 200);
            isFullscreenOnDisplay &= !editorWindow.IsInMainWin();
#endif
            EWFDebugging.LogLine("Checking isFullscreenOnDisplay: " + isFullscreenOnDisplay + " display: " + display.Bounds + " containerPosition: " + containerPosition + " containerPosPhysical: " + containerPosition, 0, 4);
            return isFullscreenOnDisplay;
        }

        /// <summary> Exit fullscreen for other windows on the screen at the specified position. Returns true if at least one fullscreen was closed. </summary>
        public static bool ExitFullscreenForOtherWindowsOnScreen(this EditorWindow editorWindow, Vector2 screenAtPosition)
        {
            bool closedAFullscreen = false;
            var allWinStates = FS.fullscreenState.window.ToArray();
            var exitingFullscreenForWinStates = new List<FS.WindowFullscreenState>();
            EditorDisplay display = EditorDisplay.ClosestToPoint(screenAtPosition);

            foreach (var win in allWinStates)
            {
                if (win.IsFullscreen && win.EditorWin != null && win.EditorWin != editorWindow && !win.currentlyRestoringFromState && win.EditorWin.IsFullscreenOnDisplay(display))
                {
                    win.EditorWin.SetFullscreen(false);
                    closedAFullscreen = true;
                    exitingFullscreenForWinStates.Add(win);
                }
            }

            return closedAFullscreen;
        }

        /// <summary> Get the EditorDisplay which currently contains the fullscreen editorWindow </summary>
        internal static EditorDisplay GetFullscreenDisplay(this EditorWindow editorWindow)
        {
            var fullscreenState = EditorFullscreenState.FindWindowState(editorWindow);
            EditorDisplay display = null;
            if (fullscreenState != null)
                display = EditorDisplay.ClosestToPoint(fullscreenState.FullscreenAtPosition);
            return display;
        }

        /// <summary> Get the window type of the editor window. (CustomSceneView and CustomGameView return their base type) </summary>
        public static Type GetWindowType(this EditorWindow editorWindow)
        {
            return FS.GetWindowType(editorWindow.GetType());
        }

        /// <summary> Get the ShowMode of the EditorWindow </summary>
        internal static ShowMode GetShowMode(this EditorWindow editorWindow)
        {
            try
            {
                var containerWindow = GetContainerWindow(editorWindow);
                FieldInfo showMode = FS.ContainerWindowType.GetField("m_ShowMode", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                return (ShowMode)showMode.GetValue(containerWindow);
            }
            catch
            {
                return ShowMode.PopupMenu;
            }
        }

        /// <summary> Set the ShowMode of the EditorWindow </summary>
        private static bool SetShowMode(this EditorWindow editorWindow, ShowMode setToMode)
        {
            try
            {
                var containerWindow = GetContainerWindow(editorWindow);
                FieldInfo showMode = FS.ContainerWindowType.GetField("m_ShowMode", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                showMode.SetValue(containerWindow, (int)setToMode);
                return true;
            }
            catch (Exception e)
            {
                EWFDebugging.LogError(e.ToString(), true);
                return false;
            }
        }

        /// <summary> Get the ContainerWindow which contains the EditorWindow </summary>
        internal static object GetContainerWindow(this EditorWindow editorWindow)
        {
            PropertyInfo window = FS.ViewType.GetProperty("window", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (window == null) EWFDebugging.LogError("Couldn't find window property of view");
            object containerWin = null;
            var hostView = editorWindow.GetHostView();
            if (hostView != null) containerWin = window.GetValue(hostView, null);
            return containerWin;
        }

        /// <summary> Get the title of the container window.</summary>
        internal static string GetContainerTitle(this EditorWindow editorWindow)
        {
            return editorWindow.GetContainerTitle(true);
        }

        /// <summary> Get the title of the container window.</summary>
        internal static string GetContainerTitle(this EditorWindow editorWindow, bool logNullContainerWarning)
        {
            var containerWin = editorWindow.GetContainerWindow();
            return GetContainerTitle(containerWin, logNullContainerWarning);
        }

        internal static string GetContainerTitle(object containerWin, bool logNullContainerWarning)
        {
            string containerTitle = null;
            if (containerWin != null)
            {
                PropertyInfo titlePI = FS.ContainerWindowType.GetProperty("title", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (titlePI != null)
                {
                    containerTitle = (string)titlePI.GetValue(containerWin, null);
                }
                else
                {
                    EWFDebugging.LogWarning("Couldn't get container title because couldn't find title property.");
                }
            }
            else
            {
                if (logNullContainerWarning) EWFDebugging.LogWarning("Couldn't get container title because couldn't find container.", true);
            }
            return containerTitle;
        }

        /// <summary> Set the title of the container window.</summary>
        internal static void SetContainerTitle(this EditorWindow editorWindow, string title)
        {
            var containerWin = editorWindow.GetContainerWindow();
            if (containerWin != null)
            {
                PropertyInfo titlePI = FS.ContainerWindowType.GetProperty("title", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (titlePI != null)
                {
                    titlePI.SetValue(containerWin, title, null);
                }
                else
                {
                    EWFDebugging.LogWarning("Couldn't set container title because couldn't find title property.");
                }
            }
            else
            {
                EWFDebugging.LogWarning("Couldn't set container title because couldn't find container.");
            }
        }

        /// <summary> Get the window position and size of the Container Window which contains the EditorWindow </summary>
        public static Rect GetContainerPosition(this EditorWindow editorWindow)
        {
            return GetContainerPosition(editorWindow, false);
        }
        /// <summary> Get the window position and size of the Container Window which contains the EditorWindow </summary>
        public static Rect GetContainerPosition(this EditorWindow editorWindow, bool logWarnings)
        {
            var containerWindow = GetContainerWindow(editorWindow);
            PropertyInfo containerPosition = FS.ContainerWindowType.GetProperty("position", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (editorWindow == null || (containerWindow == null || containerPosition == null))
            {
                if (containerWindow != null && (logWarnings == true || containerPosition == null))
                { //Before the editorWindow is shown it can have no container window, so no need to print warning here unless containerPositionProp is null.
                    EWFDebugging.LogWarning("Couldn't get container position. Falling back to EditorWindow position. containerPositionProp: " + (containerPosition == null ? "null" : "not null") + "; containerWindow: " + (containerWindow == null ? "null" : "not null"));
                }
                return editorWindow.position;
            }
            else
            {
                try
                {
                    return (Rect)containerPosition.GetValue(containerWindow, null);
                }
                catch (Exception e) { EWFDebugging.LogWarning("Error getting container position.\n" + e); }
                { return editorWindow.position; }
            }
        }
        /// <summary> Set the window position and size of the Container Window which contains the EditorWindow </summary>
        public static void SetContainerPosition(this EditorWindow editorWindow, Rect positionToSet)
        {
            var containerWindow = GetContainerWindow(editorWindow);
            PropertyInfo containerPosition = FS.ContainerWindowType.GetProperty("position", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (containerWindow == null || containerPosition == null)
            {
                EWFDebugging.LogWarning("Couldn't set container position. containerPositionProp: " + (containerPosition == null ? "null" : "not null") + "; containerWindow: " + (containerWindow == null ? "null" : "not null"));
            }
            else
            {
                containerPosition.SetValue(containerWindow, positionToSet, null);
            }
        }

        /// <summary> Get the screen point of the relative point in the Container Window of the Editor Window </summary>
        internal static Vector2 ContainerWindowPointToScreenPoint(this EditorWindow editorWindow, Vector2 windowPoint)
        {
            var containerWindow = GetContainerWindow(editorWindow);
            MethodInfo windowToScreenPoint = FS.ContainerWindowType.GetMethod("WindowToScreenPoint", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            return (Vector2)windowToScreenPoint.Invoke(containerWindow, new object[] { windowPoint });
        }

        /// <summary> Get the HostView which contains the EditorWindow </summary>
        internal static object GetHostView(this EditorWindow editorWindow)
        {
            try
            {
                FieldInfo parent = typeof(EditorWindow).GetField("m_Parent", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                return parent.GetValue(editorWindow);
            }
            catch
            {
                EWFDebugging.LogError("Couldn't find EditorWindow parent.");
                return null;
            }
        }

        /// <summary> Get the DockArea which contains the EditorWindow (if one exists) </summary>
        internal static object GetDockArea(this EditorWindow editorWindow)
        {
            try
            {
                var dockArea = GetHostView(editorWindow);
                if (dockArea.GetType() != FS.DockAreaType && dockArea.GetType().BaseType != FS.DockAreaType) dockArea = null;
                return dockArea;
            }
            catch (Exception e)
            {
                EWFDebugging.LogError("Couldn't find a DockArea for the EditorWindow.", e);
                return null;
            }
        }

        /// <summary> Returns true if the EditorWindow is docked </summary>
        internal static bool IsDocked(this EditorWindow editorWindow)
        {
            try
            {
                PropertyInfo prop = typeof(EditorWindow).GetProperty("docked", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                return (bool)prop.GetValue(editorWindow, null);
            }
            catch
            {
                return false;
            }
        }

        /// <summary> Returns true if the EditorWindow has the focus within its dock </summary>
        internal static bool HasFocusInDock(this EditorWindow editorWindow)
        {
            try
            {
                PropertyInfo prop = typeof(EditorWindow).GetProperty("hasFocus", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                return (bool)prop.GetValue(editorWindow, null);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Gets the currently docked game view if it exists and is currently focused.
        /// </summary>
        private static EditorWindow GetDockedGameView(EditorWindow excludeWindowFromSearch, bool onlyGetFocusedGameView)
        {
            var gameViews = (EditorWindow[])Resources.FindObjectsOfTypeAll(FS.GameViewType);
            foreach (var win in gameViews)
            {
                if (win != excludeWindowFromSearch)
                {
                    var winTitle = win.GetWindowTitle();
                    var containerTitle = win.GetContainerTitle();
                    bool isFSwin = winTitle.Contains("FULLSCREEN_") || (containerTitle != null && containerTitle.Contains("FULLSCREEN_"));
                    if (!isFSwin && !winTitle.Contains("TEMP_") && win.IsDocked() && (!onlyGetFocusedGameView || win.HasFocusInDock()))
                    {
                        return win;
                    }
                }
            }
            return null;
        }

        /// <summary> Make the Editor Window save to the Window Layout </summary>
        internal static void SetSaveToLayout(this EditorWindow editorWindow, bool saveToLayout)
        {
            try
            {
                FieldInfo dontSaveToLayout = FS.ContainerWindowType.GetField("m_DontSaveToLayout", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                var containerWin = editorWindow.GetContainerWindow();
                if (containerWin != null)
                    dontSaveToLayout.SetValue(containerWin, !saveToLayout);
                else if (EWFDebugging.Enabled) Debug.LogError("Container Window is null");
            }
            catch (System.Exception e)
            {
                if (EWFDebugging.Enabled)
                {
                    Debug.LogException(e);
                    EWFDebugging.LogError(e.Message);
                }
            }
        }
        internal static bool GetSaveToLayout(this EditorWindow editorWindow)
        {
            try
            {
                FieldInfo dontSaveToLayout = FS.ContainerWindowType.GetField("m_DontSaveToLayout", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                return (bool)dontSaveToLayout.GetValue(editorWindow.GetContainerWindow());
            }
            catch (System.Exception e)
            {
                if (EWFDebugging.Enabled)
                {
                    Debug.LogException(e);
                    EWFDebugging.LogError("Couldn't get dontSaveToLayout field. " + e.Message);
                }
            }
            return false;
        }

        internal static string GetWindowTitle(this EditorWindow editorWindow)
        {
            return editorWindow.titleContent.text;
        }

        internal static void SetWindowTitle(this EditorWindow editorWindow, string title)
        {
            SetWindowTitle(editorWindow, title, false);
        }

        internal static void SetWindowTitle(this EditorWindow editorWindow, string title, bool clearIcon)
        {
            if (clearIcon)
                editorWindow.titleContent = new GUIContent(title);
            else
                editorWindow.titleContent.text = title;
        }

    }
}