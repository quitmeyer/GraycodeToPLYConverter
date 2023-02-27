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


namespace EditorWindowFullscreen
{
    /// <summary>
    /// Allows changing properties of the Editor's Main Window
    /// </summary>
    public static class EditorMainWindow
    {
        internal static float topToolbarHeight = 20f;
        internal static System.Type windowLayoutType;

        private static FieldInfo containerMainViewField;
        private static PropertyInfo containerMainView;
        private static PropertyInfo containerPosition;
        private static MethodInfo containerShow;
        internal static MethodInfo containerClose;

#pragma warning disable 0414
        private static MethodInfo setPosition;
        private static MethodInfo setMinMaxSizes;
        private static PropertyInfo windowPosition;
        private static MethodInfo containerSetMinMaxSizes;
        private static MethodInfo containerCloseWin;
        private static MethodInfo containerMoveInFrontOf;
        private static MethodInfo containerMoveBehindOf;
#pragma warning restore 0414

        static EditorMainWindow()
        {
            windowLayoutType = System.Type.GetType("UnityEditor.WindowLayout,UnityEditor");

            setPosition = FS.MainWindowType.GetMethod("SetPosition", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, new[] { typeof(Rect) }, null);
            setMinMaxSizes = FS.ViewType.GetMethod("SetMinMaxSizes", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, new[] { typeof(Vector2), typeof(Vector2) }, null);
            windowPosition = FS.ViewType.GetProperty("windowPosition", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            containerMainViewField = FS.ContainerWindowType.GetField("m_MainView", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (containerMainViewField == null)
                containerMainViewField = FS.ContainerWindowType.GetField("m_RootView", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            containerMainView = FS.ContainerWindowType.GetProperty("mainView", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (containerMainView == null)
                containerMainView = FS.ContainerWindowType.GetProperty("rootView", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            containerPosition = FS.ContainerWindowType.GetProperty("position", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            containerSetMinMaxSizes = FS.ContainerWindowType.GetMethod("SetMinMaxSizes", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, new[] { typeof(Vector2), typeof(Vector2) }, null);
            containerShow = FS.ContainerWindowType.GetMethod("Show", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, new[] { typeof(int), typeof(bool), typeof(bool) }, null);
            if (containerShow == null) if (containerShow == null) containerShow = FS.ContainerWindowType.GetMethod("Show", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, new[] { Type.GetType("UnityEditor.ShowMode,UnityEditor"), typeof(bool), typeof(bool) }, null);
            if (containerShow == null) containerShow = FS.ContainerWindowType.GetMethod("Show", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, new[] { Type.GetType("UnityEditor.ShowMode,UnityEditor"), typeof(bool), typeof(bool), typeof(bool) }, null);

            containerClose = FS.ContainerWindowType.GetMethod("Close", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, new System.Type[] { }, null);
            containerCloseWin = FS.ContainerWindowType.GetMethod("InternalCloseWindow", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, new System.Type[] { }, null);
            containerMoveInFrontOf = FS.ContainerWindowType.GetMethod("MoveInFrontOf", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, new System.Type[] { FS.ContainerWindowType }, null);
            containerMoveBehindOf = FS.ContainerWindowType.GetMethod("MoveBehindOf", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, new System.Type[] { FS.ContainerWindowType }, null);
        }

        internal static object FindMainWindow()
        {
            try
            {
                object mainWin = null;

                var containers = Resources.FindObjectsOfTypeAll(FS.ContainerWindowType);
                foreach (var container in containers)
                {
                    var rootView = containerMainView.GetValue(container, null);
                    if (rootView != null && rootView.GetType() == FS.MainWindowType)
                    {
                        mainWin = rootView;
                        break;
                    }
                }

                return mainWin;
            }
            catch
            {
                Debug.LogError("Couldn't find the editor Main Window");
                return null;
            }
        }

        internal static object FindContainerWindow()
        {
            var mainWindow = FindMainWindow();
            if (mainWindow == null)
                return null;
            else
            {
                try
                {
                    PropertyInfo window = FS.ViewType.GetProperty("window", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                    return window.GetValue(mainWindow, null);
                }
                catch
                {
                    Debug.LogError("Couldn't find the editor Container Window");
                    return null;
                }
            }
        }

        internal static object FindOriginalContainerWindow()
        {
            return FindOriginalContainerWindow(null);
        }
        internal static object FindOriginalContainerWindow(object destroyAllMainWindowsExceptThis)
        {
            var containers = Resources.FindObjectsOfTypeAll(FS.ContainerWindowType);

            foreach (var container in containers)
            {
                EditorWindowExtensions.ShowMode showMode;
                try
                {
                    PropertyInfo showModeProp = FS.ContainerWindowType.GetProperty("showMode", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                    showMode = (EditorWindowExtensions.ShowMode)showModeProp.GetValue(container, null);
                    if (showMode == EditorWindowExtensions.ShowMode.MainWindow)
                    {
                        if (destroyAllMainWindowsExceptThis == null || (UnityEngine.Object)destroyAllMainWindowsExceptThis == container)
                        {
                            return container;
                        }
                        else
                        {
                            if (containerClose != null)
                                containerClose.Invoke(container, null);
                        }
                    }
                }
                catch (System.Exception e)
                {
                    Debug.LogError("Failed to find the Original Container Window (Error retrieving showMode property) " + e.Message);
                }
            }
            return null;
        }

        internal static IntPtr GetWindowHandle()
        {
            try
            {
                FieldInfo winPtr = FS.ContainerWindowType.GetField("m_WindowPtr", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                object wPtr = winPtr.GetValue(FindContainerWindow());
                IntPtr ptr = (IntPtr)wPtr.GetType().GetField("m_IntPtr", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance).GetValue(wPtr);
                return ptr;
            }
            catch
            {
                Debug.LogError("Couldn't find the editor main window handle.");
                return IntPtr.Zero;
            }
        }

        public static Rect position
        {
            get
            {
                PropertyInfo position = FS.ViewType.GetProperty("screenPosition", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                return (Rect)position.GetValue(FindMainWindow(), null);
            }
            set
            {
                containerPosition.SetValue(FindContainerWindow(), value, null);
            }
        }

        public static Vector2 minSize
        {
            get
            {
                PropertyInfo minSize = FS.ViewType.GetProperty("minSize", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                return (Vector2)minSize.GetValue(FindMainWindow(), null);
            }
        }

        public static Vector2 maxSize
        {
            get
            {
                PropertyInfo maxSize = FS.ViewType.GetProperty("maxSize", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                return (Vector2)maxSize.GetValue(FindMainWindow(), null);
            }
        }

        public static bool maximized
        {
            get
            {
                //PropertyInfo maximized = FS.containerWindowType.GetProperty("maximized", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                //return (bool)maximized.GetValue(FindContainerWindow(), null);
                var pos = position;
                var pixelsPerPoint = GetPixelsPerPointAtPosition(pos.center);
                var displayBounds = EditorDisplay.ClosestToPoint(pos.center).Bounds;
                return pos.x == displayBounds.x && pos.y <= displayBounds.y + 50 && pos.width * pixelsPerPoint == displayBounds.width && Mathf.Abs(displayBounds.height - pos.height * pixelsPerPoint) <= 200;
            }
        }

        private static float GetPixelsPerPointAtPosition(Vector2 atPosition)
        {
            var screenBounds = EditorDisplay.ClosestToPoint(atPosition).Bounds;

            //Use a dummy scene view to get the real pixels per point (Because EditorGUIUtility.pixelsPerPoint is inconsistent).
            var dummySceneView = EditorWindowExtensions.CreateWindow(typeof(SceneView));
            dummySceneView.SetFullscreen(true, atPosition, false);
            var pixelsPerPoint = screenBounds.width / dummySceneView.position.width;
            dummySceneView.SetFullscreen(false);
            if (dummySceneView != null) dummySceneView.Close();
            if (dummySceneView != null) UnityEngine.Object.DestroyImmediate(dummySceneView);
            return pixelsPerPoint;
        }

        public static void ToggleMaximize()
        {
            MethodInfo toggleMaximize = FS.ContainerWindowType.GetMethod("ToggleMaximize", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, new System.Type[] { }, null);
            toggleMaximize.Invoke(FindContainerWindow(), null);
        }

        public static void SetMinMaxSizes(Vector2 minSize, Vector2 maxSize)
        {
            SetMinMaxSizes(FindMainWindow(), minSize, maxSize);
        }

        public static void SetMinMaxSizes(object viewObj, Vector2 minSize, Vector2 maxSize)
        {
            setMinMaxSizes.Invoke(viewObj, new object[] { minSize, maxSize });
        }

        public static void Focus()
        {
            var win = ScriptableObject.CreateInstance<EditorWindow>();
            win.Show(true);
            win.Focus();
            win.Close();
        }

        internal static void TriggerOnResizedAll()
        {
            var containerWin = FindContainerWindow();
            if (containerWin != null)
            {
                MethodInfo OnResizedC = FS.ContainerWindowType.GetMethod("OnResize", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (OnResizedC != null) OnResizedC.Invoke(containerWin, null);
                else EWFDebugging.LogWarning("ContainerWindow OnResize method doesn't exist.");
            }
        }

        public static FS.WindowFullscreenState GetWindowFullscreenState()
        {
            return FS.FindWindowState(null, FS.MainWindowType);
        }

        public static bool IsFullscreen()
        {
            var fullscreenState = GetWindowFullscreenState();
            return fullscreenState.IsFullscreen;
        }

        public static bool IsFullscreenAtPosition(Vector2 checkPosition)
        {
            var currentScreenBounds = EditorDisplay.ClosestToPoint(position.center).Bounds;
            var fullscreenBoundsAtCheckPosition = EditorDisplay.ClosestToPoint(checkPosition).Bounds;
            var fullscreenState = GetWindowFullscreenState();
            return fullscreenState.IsFullscreen && currentScreenBounds == fullscreenBoundsAtCheckPosition;
        }

        public static bool ToggleFullscreen()
        {
            return ToggleFullscreen(true);
        }
        public static bool ToggleFullscreen(bool showTopToolbar)
        {
            return ToggleFullscreen(showTopToolbar, position.center);
        }
        public static bool ToggleFullscreen(bool showTopToolbar, Vector2 fullscreenAtPosition)
        {
            var fullscreenState = GetWindowFullscreenState();

#if UNITY_EDITOR_OSX
            bool setFullscreen = !SystemDisplay.MainWindowIsFullscreen();
#else
            var currentScreenBounds = EditorDisplay.ClosestToPoint(position.center).Bounds;
            var newFullscreenBounds = EditorDisplay.ClosestToPoint(fullscreenAtPosition).Bounds;
            bool setFullscreen = (!fullscreenState.IsFullscreen || currentScreenBounds != newFullscreenBounds);
#endif

            if (EditorWindowExtensions.ExitFullscreenForOtherWindowsOnScreen(fullscreenState.EditorWin, fullscreenAtPosition))
            {
                setFullscreen = true;
                SetFullscreen(setFullscreen, showTopToolbar, fullscreenAtPosition);
            }

            SetFullscreen(setFullscreen, showTopToolbar, fullscreenAtPosition);
            return setFullscreen;
        }

        public static void SetFullscreen(bool fullscreen)
        {
            SetFullscreen(fullscreen, true);
        }

        public static void SetFullscreen(bool fullscreen, bool showTopToolbar)
        {
            SetFullscreen(fullscreen, showTopToolbar, position.center);
        }

        public static void SetFullscreen(bool fullscreen, bool showTopToolbar, Vector2 fullscreenAtPosition)
        {
            SetFullscreen(fullscreen, showTopToolbar, fullscreenAtPosition, false);
        }
        public static void SetFullscreen(bool fullscreen, bool showTopToolbar, Vector2 fullscreenAtPosition, bool disableUpdatePrePos)
        {
            var fullscreenState = GetWindowFullscreenState();
            var fullscreenOnDisplay = EditorDisplay.ClosestToPoint(fullscreenAtPosition);
            var screenBounds = fullscreenOnDisplay.Bounds;

#if UNITY_EDITOR_OSX
            if (fullscreenOnDisplay.Locked)
            {
                FS.RunAfterDisplayNotLocked(fullscreenAtPosition, () => SetFullscreen(fullscreen, showTopToolbar, fullscreenAtPosition, disableUpdatePrePos));
                return;
            }
#endif

            var originallyFocusedEditorWin = EditorWindow.focusedWindow;
            var originallyFocusedEditorWinType = originallyFocusedEditorWin == null ? null : originallyFocusedEditorWin.GetType();
            object mainWindow = null;

#if UNITY_EDITOR_OSX
            bool wasFullscreen = SystemDisplay.MainWindowIsFullscreen(); //If toggling the top toolbar, don't update pre positions.
            bool updatePrePos = fullscreen && !wasFullscreen && !disableUpdatePrePos;
            windowController = IntPtr.Zero;
            if (fullscreen)
            {
                fullscreenState.ScreenBounds = screenBounds;
                fullscreenState.FullscreenAtPosition = fullscreenAtPosition;
            }
            if (updatePrePos)
            {
                fullscreenState.PreFullscreenPosition = position;
                fullscreenState.PreFullscreenMinSize = minSize;
                fullscreenState.PreFullscreenMaxSize = maxSize;
            }

            var prePos = fullscreenState.PreFullscreenPosition;
            if (prePos.width < 100 || prePos.height < 100 || prePos.width < fullscreenState.PreFullscreenMinSize.x || prePos.height < fullscreenState.PreFullscreenMinSize.y)
                prePos = new Rect(prePos.x, prePos.y, Mathf.Max(fullscreenState.PreFullscreenMinSize.x, 300), Mathf.Max(fullscreenState.PreFullscreenMinSize.y, 300)); //Make sure size is valid

            if (fullscreen && !screenBounds.Contains(position.center))
            {
                if (wasFullscreen)
                {
                    //Exit fullscreen because we are fullscreen on another screen
                    fullscreen = false;
                }
                else
                {
                    //Move to the correct screen
                    SystemDisplay.SetMainWindowPosition((int)screenBounds.xMin, (int)screenBounds.yMin, (int)screenBounds.width, (int)screenBounds.height);
                }
            }

            if (fullscreen != wasFullscreen)
            {
                windowController = SystemDisplay.ToggleFullscreenMainWindow((int)prePos.xMin, (int)prePos.yMin, (int)prePos.width, (int)prePos.height);

                if (!fullscreen)
                {
                    if (fullscreenState.ScreenBounds.Contains(prePos))
                    {
                        position = prePos; //Setting the position here first (even though still haven't finished exiting fullscreen) updates the docked window sizes so the shrinking animation is smoother.
                    } else
                    {
                        var intermediatePos = fullscreenState.PreFullscreenPosition.CenterRectInBounds(fullscreenState.ScreenBounds);
                        position = intermediatePos; //Can't move screen yet because still fullscreen, so use an intermediate pos.
                    }

                    //Restore position once the fullscreen has finished exiting
                    if (windowController != IntPtr.Zero)
                    {
                        fullscreenOnDisplay.Locked = true;
                        EditorApplication.update += CheckForFinishExitingFullscreen;
                        numChecksForFinishExiting = 0;
#if UNITY_2019_3_OR_NEWER
                        TriggerOnResizedAll();
#endif
                    }
                } else
                {
#if UNITY_2019_3_OR_NEWER
                    TriggerOnResizedAll();
#endif
                    fullscreenOnDisplay.Locked = true;
                    FS.RunAfter(() => { return false; }, () => 
                    {
                        fullscreenOnDisplay.Locked = false;
                        fullscreenState.currentlyRestoringFromState = false;
                        FS.SaveFullscreenState();
                        TriggerOnResizedAll();
                    }, 50, true);
                }
            }

            fullscreenState.IsFullscreen = fullscreen;
            FS.SaveFullscreenState();
#else

            bool wasFullscreen = fullscreenState.IsFullscreen;
            fullscreenState.ShowTopToolbar = showTopToolbar;
            fullscreenState.originalContainerWindow = (ScriptableObject)FindOriginalContainerWindow();
            fullscreenState.containerWindow = (ScriptableObject)FindContainerWindow();
            mainWindow = FindMainWindow();

#if UNITY_2018_2_OR_NEWER
            var pixelsPerPoint = GetPixelsPerPointAtPosition(fullscreenAtPosition);
            screenBounds.width /= pixelsPerPoint;
            screenBounds.height /= pixelsPerPoint;
#endif

            if (fullscreen)
            {
                fullscreenState.ScreenBounds = screenBounds;
                fullscreenState.FullscreenAtPosition = fullscreenAtPosition;

                if (!wasFullscreen)
                {
                    var wasMaximized = maximized;
                    if (wasMaximized)
                        ToggleMaximize();

                    fullscreenState.PreFullscreenPosition = position;
                    fullscreenState.PreFullscreenMinSize = minSize;
                    fullscreenState.PreFullscreenMaxSize = maxSize;
                    fullscreenState.PreFullscreenMaximized = wasMaximized;

                    SystemDisplay.SaveMainWindowStyleInState(fullscreenState);
                }
            }

            if (fullscreen && !showTopToolbar)
            {
                var newPos = screenBounds;
                newPos.yMin += topToolbarHeight;
                position = newPos;
                SetMinMaxSizes(newPos.size, newPos.size);
                position = newPos;
                SystemDisplay.SetMainWindowFullscreenStyle(fullscreenState, showTopToolbar);
                fullscreenState.IsFullscreen = true;
            }
            else
            {
                SystemDisplay.SetMainWindowFullscreenStyle(fullscreenState, showTopToolbar);

                if (fullscreenState.EditorWin != null)
                    fullscreenState.EditorWin.Close();

                if (fullscreen)
                {
                    //Set fullscreen with toolbar
                    var newPos = screenBounds;
                    newPos.yMin += topToolbarHeight;

                    position = newPos;
                    SetMinMaxSizes(newPos.size, newPos.size);
                    position = newPos;


                    if (position.x != newPos.x)
                    {
                        //Position didn't set correctly, so must be maximized
                        ToggleMaximize();
                        position = newPos;
                    }

                    fullscreenState.IsFullscreen = true;
                }
            }

            if (fullscreen)
            {
                MethodInfo displayAllViews = FS.ContainerWindowType.GetMethod("DisplayAllViews", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, new System.Type[] { }, null);
                if (displayAllViews != null) displayAllViews.Invoke(fullscreenState.containerWindow, null);
                TriggerOnResizedAll();
            }


            if (!fullscreen && wasFullscreen && mainWindow != null)
            {
                //Reset position
                var prePos = fullscreenState.PreFullscreenPosition;
                position = prePos;
                fullscreenState.IsFullscreen = false;
                position = fullscreenState.PreFullscreenPosition;
                SetMinMaxSizes(fullscreenState.PreFullscreenMinSize, fullscreenState.PreFullscreenMaxSize);
                position = fullscreenState.PreFullscreenPosition;
                PropertyInfo pos = FS.ViewType.GetProperty("position", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                pos.SetValue(mainWindow, fullscreenState.PreFullscreenPosition, null);

                fullscreenState.IsFullscreen = false;

                SystemDisplay.LoadMainWindowStyleInState(fullscreenState, true);
                TriggerOnResizedAll();

                Focus();

                position = fullscreenState.PreFullscreenPosition; //Reset position
                position = fullscreenState.PreFullscreenPosition;

                if (fullscreenState.PreFullscreenMaximized != maximized)
                    ToggleMaximize();
            }

            fullscreenState.currentlyRestoringFromState = false;
            FS.SaveFullscreenState();
#endif
            //All platforms
            FS.TriggerFullscreenEvent(mainWindow, FS.MainWindowType, fullscreenAtPosition, fullscreen);
            if (EditorWindow.focusedWindow == null)
            {
                if (originallyFocusedEditorWin != null)
                    originallyFocusedEditorWin.Focus();
                else if (originallyFocusedEditorWinType != null)
                {
                    EditorWindow.FocusWindowIfItsOpen(originallyFocusedEditorWinType);
                }
            }
        }
#if UNITY_EDITOR_OSX
        private static IntPtr windowController = IntPtr.Zero;
        private static int numChecksForFinishExiting = 0;
        private static void CheckForFinishExitingFullscreen()
        {
            numChecksForFinishExiting++;
            if (windowController == IntPtr.Zero || numChecksForFinishExiting > 500)
                EditorApplication.update -= CheckForFinishExitingFullscreen;
            else
            {
                try
                {
                    var isStillExitingFullscreen = SystemDisplay.MainWindowIsExitingFullscreen();
                    if (!isStillExitingFullscreen)
                    {
                        EditorApplication.update -= CheckForFinishExitingFullscreen;

                        //Finished exiting fullscreen, so reset the window position.
                        var fullscreenState = GetWindowFullscreenState();
                        var fullscreenOnDisplay = EditorDisplay.ClosestToPoint(fullscreenState.FullscreenAtPosition);
                        fullscreenOnDisplay.Locked = false;
                        var prePos = fullscreenState.PreFullscreenPosition;
                        prePos.width = Mathf.Max(300, prePos.width);
                        prePos.height = Mathf.Max(300, prePos.height);
                        position = prePos;
                        fullscreenState.currentlyRestoringFromState = false;
                        FS.SaveFullscreenState();
                        TriggerOnResizedAll();
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError(e);
                    EditorApplication.update -= CheckForFinishExitingFullscreen;
                }
            }
        }
#endif
        internal static object[] GetAllChildViews()
        {
            try
            {
                var mainWindow = FindMainWindow();
                PropertyInfo allChildren = FS.ViewType.GetProperty("allChildren", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                return (object[])allChildren.GetValue(mainWindow, null);
            }
            catch (System.Exception e)
            {
                if (EWFDebugging.Enabled)
                {
                    Debug.LogException(e);
                    EWFDebugging.LogError(e.Message);
                }
            }
            return null;
        }
    }
}