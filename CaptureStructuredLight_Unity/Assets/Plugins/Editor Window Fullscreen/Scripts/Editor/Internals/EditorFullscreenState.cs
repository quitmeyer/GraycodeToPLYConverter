/* 
 * Author:  Johanan Round
 * Package: Editor Window Fullscreen
 * License: Unity Asset Store EULA (Editor extension asset. Requires 1 license per machine.)
 */
using System.Xml.Serialization;
using UnityEngine;
using UnityEditor;

using System;
using System.IO;
using System.Reflection;
using System.Reflection.Emit;
using System.Collections.Generic;

using System.CodeDom;
using System.Linq.Expressions;
using System.Runtime;

namespace EditorWindowFullscreen
{
    /// <summary>
    /// Stores the current fullscreen state, and provides methods for modifying it.
    /// </summary>
    public class EditorFullscreenState
    {
        internal delegate void FullscreenEvent(object window, Type windowType, Vector2 atPosition, bool enteredFullscreen);
        internal static event FullscreenEvent FullscreenEventHandler;

        internal static float topTabFullHeight = 39; //Initialized as a fallback. It is calculated automatically in SetBorderlessPosition().
        internal static float windowTopPadding = 5; //^^
        internal static bool calculatedBorderlessOffsets;
#if UNITY_2019_3_OR_NEWER
        internal const float sceneViewToolbarHeight = 21f;
#else
        internal const float sceneViewToolbarHeight = 17f;
#endif
        internal static string projectLibraryPath;
        internal static System.Type ViewType { get { return Type.GetType("UnityEditor.View,UnityEditor"); } }
        internal static System.Type HostViewType { get { return Type.GetType("UnityEditor.HostView,UnityEditor"); } }
        internal static System.Type DockAreaType { get { return Type.GetType("UnityEditor.DockArea,UnityEditor"); } }
        internal static System.Type ContainerWindowType { get { return Type.GetType("UnityEditor.ContainerWindow,UnityEditor"); } }

        internal static System.Type MainWindowType
        {
            get
            {
                var mainWindowType = Type.GetType("UnityEditor.MainWindow,UnityEditor");
                if (mainWindowType == null) mainWindowType = System.Type.GetType("UnityEditor.MainView,UnityEditor");
                return mainWindowType;
            }
        }
        internal static System.Type SceneViewType { get { return typeof(SceneView); } }
        internal static System.Type GameViewType { get { return Type.GetType("UnityEditor.GameView,UnityEditor"); } }
#if UNITY_2019_3_OR_NEWER
        internal static System.Type PlayModeViewType { get { return Type.GetType("UnityEditor.PlayModeView,UnityEditor"); } }
#endif
        internal static System.Type ConsoleWindowType { get { return Type.GetType("UnityEditor.ConsoleWindow,UnityEditor"); } }
        internal static System.Type InspectorWindowType { get { return Type.GetType("UnityEditor.InspectorWindow,UnityEditor"); } }
        internal static System.Type DisplayUtilityType { get { return Type.GetType("UnityEditor.DisplayUtility,UnityEditor"); } }
        internal static System.Type FlexibleMenuType { get { return Type.GetType("UnityEditor.FlexibleMenu,UnityEditor"); } }
        internal static System.Type GameViewSizesType { get { return Type.GetType("UnityEditor.GameViewSizes,UnityEditor"); } }
        internal static System.Type GameViewSizesMenuItemProvider { get { return Type.GetType("UnityEditor.GameViewSizesMenuItemProvider,UnityEditor"); } }
        internal static System.Type GameViewSizesMenuModifyItemUI { get { return Type.GetType("UnityEditor.GameViewSizesMenuModifyItemUI,UnityEditor"); } }

        [Serializable]
        public class WindowFullscreenState
        {
            public bool IsFullscreen;
            public bool ShowTopTabs;
            public bool ShowTopToolbar;
            public bool CloseOnExitFullscreen;
            public string WindowName;
            public string WindowTitle;
            public IntPtr WindowHandle;

            [SerializeField]
            public string actualTypeAssemblyQualifiedName; //Must be public for XmlSerializer
            private Type actualType;
            [XmlIgnore]
            public Type ActualType
            {
                get { return actualType; }
                set
                {
                    actualType = value;
                    if (actualType != null)
                        actualTypeAssemblyQualifiedName = actualType.AssemblyQualifiedName;
                    else
                        actualTypeAssemblyQualifiedName = null;
                }
            }

            [SerializeField]
            public string windowTypeAssemblyQualifiedName; //Must be public for XmlSerializer
            private Type windowType;
            [XmlIgnore]
            public Type WindowType
            {
                get { return windowType; }
                set
                {
                    windowType = value;
                    if (windowType != null)
                        windowTypeAssemblyQualifiedName = windowType.AssemblyQualifiedName;
                    else
                        windowTypeAssemblyQualifiedName = null;
                }
            }
            [XmlIgnore]
            public EditorWindow EditorWin;
            [XmlIgnore]
            public ScriptableObject containerWindow;
            [XmlIgnore]
            public ScriptableObject originalContainerWindow;
            public Rect PreFullscreenPosition;
            public Vector2 PreFullscreenMinSize;
            public Vector2 PreFullscreenMaxSize;
            public bool PreFullscreenMaximized;
            public CursorLockMode CursorLockModePreShowTopToolbar;
            public Rect ScreenBounds;
            public Vector2 FullscreenAtPosition;
            public Rect ContainerPosition;
            public bool HasFocus;
            public bool CreatedAtGameStart;
            public bool UnfocusedGameViewOnEnteringFullscreen;
            public bool CreatedNewWindow;
            public int OriginalStyle;
            public int OriginalExStyle;
            public int fullscreenOptionID;
            public bool initialOptionsWereSet;
            public bool fullscreenPositionWasSet;
            public long identifierID;
            public string identifierTitle;
            public float serializedID;
            public bool currentlyRestoringFromState;
            public EditorFullscreenSettings.FullscreenOption FullscreenOptions
            {
                get { return EditorFullscreenSettings.settings.GetFullscreenOption(fullscreenOptionID); }
            }

            public void SetEditorWin(EditorWindow editorWin)
            {
                this.WindowName = editorWin.name;
                this.WindowTitle = editorWin.GetWindowTitle();
                this.EditorWin = editorWin;

                this.WindowType = editorWin.GetWindowType();
                this.ActualType = editorWin.GetType();
            }

            public WindowFullscreenState CreateCopy()
            {
                return (WindowFullscreenState)this.MemberwiseClone();
            }
        }

        [Serializable]
        public struct FullscreenState
        {
            public List<WindowFullscreenState> window;

            public void CleanDeletedWindows()
            {
                window.RemoveAll(win => win.EditorWin == null && win.containerWindow == null && win.WindowType != MainWindowType || win.ActualType == typeof(EditorInput));

                //Ensure there is only one state for the main window.
                var mainWinState = window.FindLast(win => win.WindowType == MainWindowType);
                window.RemoveAll(win => win.WindowType == MainWindowType && win != mainWinState);
            }
        }

        internal static FullscreenState fullscreenState;
        private const string FullscreenStateFilename = "CurrentEditorFullscreenWindowState.fwst";

        private static List<WindowFullscreenState> queuedStatesToToggleOnLoad = new List<WindowFullscreenState>();
        internal delegate void RunOnLoad();
        private static RunOnLoad RunOnNextLoadMethods;

        public static WindowFullscreenState FindWindowState(EditorWindow editorWin)
        {
            if (editorWin == null)
                return null;
            else
                return FindWindowState(editorWin, editorWin.GetType());
        }
        public static WindowFullscreenState FindWindowState(System.Type windowType)
        {
            return FindWindowState(null, windowType);
        }
        public static WindowFullscreenState FindWindowState(EditorWindow editorWin, System.Type windowType)
        {
            return FindWindowState(editorWin, windowType, null);
        }
        public static WindowFullscreenState FindWindowState(EditorWindow editorWin, System.Type windowType, EditorDisplay editorDisplay)
        {
            WindowFullscreenState winState = null;
            Type actualType = windowType;
            windowType = GetWindowType(actualType);
            string containerTitle = null;
            bool inFullscreenContainer = false;

            if (fullscreenState.window != null)
            {
                if (editorWin != null && !editorWin.IsInMainWin() && windowType != MainWindowType)
                {
                    //See if the window is part of an existing fullscreen container
                    containerTitle = editorWin.GetContainerTitle(false);
                    inFullscreenContainer = containerTitle != null && containerTitle.StartsWith("FULLSCREEN", StringComparison.Ordinal);
                    if (inFullscreenContainer)
                    {
                        winState = fullscreenState.window.Find(state => state.ShowTopTabs && state.identifierTitle != null && state.identifierTitle == containerTitle);
                        if (winState != null)
                        {
                            //If so, use that fullscreen state
                            winState.SetEditorWin(editorWin);
                            return winState;
                        }
                    }
                }


                try
                {
                    if (windowType == MainWindowType)
                        winState = fullscreenState.window.Find(state => state.WindowType == MainWindowType);
                    else if (editorWin != null)
                        winState = fullscreenState.window.Find(state => state.EditorWin == editorWin);
                    else
                    {
                        foreach (var state in fullscreenState.window)
                        {
                            if (state.WindowType == windowType && (state.EditorWin != null || state.containerWindow != null))
                            {
                                if (state.EditorWin != null && state.EditorWin.IsFullscreenOnDisplay(editorDisplay))
                                {
                                    winState = state;
                                    break;
                                }
                                else if (editorDisplay == null)
                                {
                                    winState = state;
                                    break;
                                }
                            }
                        }
                    }
                }
                catch (System.Exception e)
                {
                    if (EWFDebugging.Enabled)
                    {
                        string error = "Couldn't find window state. Error: " + e.ToString();
                        EWFDebugging.LogError(error);
                    }
                }
            }

            if (editorWin != null && winState != null && winState.ShowTopTabs && !inFullscreenContainer && !string.IsNullOrEmpty(winState.identifierTitle))
            {
                //We are an existing fullscreen state which has been moved out of its fullscreen container.
                var oldState = winState;
                winState = oldState.CreateCopy(); //Therefore, abandon the old state and create a new one.
                winState.identifierID = 0;
                winState.identifierTitle = null;
                winState.fullscreenPositionWasSet = false;
                winState.SetEditorWin(editorWin);
                AddWindowState(winState);
                winState.EditorWin.SetIdentifierTitle(winState); //Give a new identifier to the new container.
                winState.FullscreenAtPosition = winState.EditorWin.position.center; //We might have moved screen but IsFullscreen is still true, so must update this prop.
            }

            if (winState == null)
            {
                winState = AddWindowState(editorWin, windowType, actualType);
            }

            return winState;
        }

        internal static WindowFullscreenState lastWinStateToChangeFullscreenStatus;

        public static WindowFullscreenState FocusedWindowState
        {
            get
            {
                if (EditorWindow.focusedWindow == null)
                    return null;
                else
                    return FindWindowState(EditorWindow.focusedWindow);
            }
        }

        private static WindowFullscreenState AddWindowState(EditorWindow editorWin, Type windowType, Type actualType)
        {
            var winState = new WindowFullscreenState();

            winState.WindowType = windowType;
            winState.ActualType = actualType;

            if (editorWin != null)
            {
                winState.SetEditorWin(editorWin);
            }
            else if (windowType == MainWindowType)
            {
                winState.WindowName = MainWindowType.Name;
                winState.WindowTitle = "Unity Editor";
                winState.containerWindow = (ScriptableObject)EditorMainWindow.FindContainerWindow();
                winState.originalContainerWindow = winState.containerWindow;
            }

            AddWindowState(winState);

            return winState;
        }

        private static WindowFullscreenState AddWindowState(WindowFullscreenState winState)
        {
            if (fullscreenState.window == null)
            {
                fullscreenState.window = new List<WindowFullscreenState>();
            }

            fullscreenState.window.Add(winState);
            return winState;
        }

        private class FirstLoadSinceStartupCompletedToken : ScriptableObject { }
        private static bool FirstLoadSinceStartup
        {
            get
            {
                var token = ScriptableObject.FindObjectOfType<FirstLoadSinceStartupCompletedToken>();
                return token == null;
            }
            set
            {
                if (!value && FirstLoadSinceStartup) ScriptableObject.CreateInstance<FirstLoadSinceStartupCompletedToken>();
            }
        }

        [InitializeOnLoadMethod]
        private static void Initialize()
        {
            FullscreenEventHandler = null;
            projectLibraryPath = Directory.GetCurrentDirectory() + "/Library";

            if (ViewType == null || HostViewType == null || DockAreaType == null || MainWindowType == null || ContainerWindowType == null || GameViewType == null)
            {
                if (EWFDebugging.Enabled)
                {
                    string error = "One or more Window/View types could not be found...\n";
                    string errorLine2 = "viewType: " + TypeString(ViewType) + " hostViewType: " + TypeString(HostViewType) + " dockAreaType: " + TypeString(DockAreaType) + " mainWindowType: " + TypeString(MainWindowType) + " containerWindowType: " + TypeString(ContainerWindowType) + " gameViewType: " + TypeString(GameViewType);
                    if (errorLine2 != null) error += errorLine2;
                    Debug.LogError(error);
                    EWFDebugging.LogError(error);
                }
            }
#if UNITY_EDITOR_OSX
            SystemDisplay.EnableDebugging(EWFDebugging.Enabled);
#endif
            EditorApplication.update += InitialLoadState;
        }

        private static string TypeString(Type type)
        {
            return type == null ? "null" : type.ToString();
        }

        internal static bool loadedInitialState = false;
        private static int loadChecks = 0;
        private static double timeAtInit = 0;
        private static void InitialLoadState()
        {
            if (loadedInitialState) FinishInitialLoadState();
            if (timeAtInit == 0) timeAtInit = EditorApplication.timeSinceStartup;

            loadChecks++;

            if (loadChecks < 2000)
            {
                var allWins = Resources.FindObjectsOfTypeAll<EditorWindow>();
                bool gameIsStartingOrStopping = (EditorApplication.isPlayingOrWillChangePlaymode != EditorApplication.isPlaying);
                float minWaitTime = EditorApplication.timeSinceStartup < 20 ? 1 : 0.1f;
                if (allWins.Length < 1 || EditorApplication.isCompiling || gameIsStartingOrStopping || (EditorApplication.timeSinceStartup - timeAtInit < minWaitTime))
                {
                    //Wait for the editor windows to be loaded before attempting to load fullscreen state.
                    return;
                }
            }
            try
            {
                LoadFullscreenState();
            }
            catch (System.Exception e)
            {
                EWFDebugging.LogError("Error loading fullscreen state. " + e.Message + "\n" + e.StackTrace + "\n\n" + (e.InnerException != null ? e.InnerException.ToString() : ""), true);
            }
            FinishInitialLoadState();
        }
        private static void FinishInitialLoadState()
        {
            loadedInitialState = true;
            EditorApplication.update -= InitialLoadState;
#if UNITY_EDITOR_OSX
            EditorApplication.update += FullscreenValidator;
#endif
            FirstLoadSinceStartup = false;
        }
#if UNITY_EDITOR_OSX
        private static List<WindowFullscreenState> validateWins;
        private static long validateCount = 0;
        private static void FullscreenValidator()
        {
            if (validateWins != null && validateWins.Count > 0)
            {
                validateCount++;
                if (validateCount % 10 != 0) return;
                for (int i = 0; i < validateWins.Count; i++)
                {
                    var s = validateWins[i];
                    if ((s.IsFullscreen && !s.ShowTopTabs && !s.ShowTopToolbar && s.EditorWin != null && s.WindowType == GameViewType) == false) continue; //Check that the state still should be validated.

                    var editorWindow = validateWins[i].EditorWin;
                    object hostView = editorWindow.GetHostView();
                    if (hostView != null)
                    {
                        var viewPos = ViewType.GetProperty("position", BindingFlags.Public | BindingFlags.Instance | BindingFlags.NonPublic);
                        if (viewPos != null)
                        {
                            var currentViewPos = (Rect)viewPos.GetValue(hostView, null);
                            if (currentViewPos.y >= 0)
                            {
                                var fullscreenOnDisplay = EditorDisplay.ClosestToPoint(validateWins[i].FullscreenAtPosition);
                                if (!fullscreenOnDisplay.Locked)
                                {
                                    //Toolbar isn't hidden but it should be, so hide it now.
                                    editorWindow.SetToolbarVisibilityAtPos(validateWins[i].ScreenBounds, true, false);
                                } else
                                {
                                    validateCount = 9; //Display is locked, so check again next update.
                                }
                            }
                        }
                    }
                }
            }
        }
#endif

        internal static void SaveFullscreenState()
        {
            try
            {
                fullscreenState.CleanDeletedWindows();

                float serializedID = 5000f;
                //Update state window container positions and focus before saving
                foreach (var state in fullscreenState.window)
                {
                    serializedID += 1f;
                    if (state.EditorWin != null)
                    {
                        state.ContainerPosition = state.EditorWin.GetContainerPosition();
                        state.HasFocus = state.EditorWin == EditorWindow.focusedWindow;

                        if (state.identifierID != 0)
                        {
                            var container = state.EditorWin.GetContainerWindow();
                            if (container != null)
                            {
                                var containerMaxSize = ContainerWindowType.GetField("m_MaxSize", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                                if (containerMaxSize != null)
                                {
                                    containerMaxSize.SetValue(container, new Vector2(serializedID, serializedID));
                                    state.serializedID = serializedID;
                                }
                            }
                        }
                    }
                }

#if UNITY_EDITOR_OSX
                //Update windows to validate
                if (validateWins != null) validateWins.Clear();
                validateWins = fullscreenState.window.FindAll(s => s.IsFullscreen && !s.ShowTopTabs && !s.ShowTopToolbar && s.EditorWin != null && s.WindowType == GameViewType);
#endif

                string fullscreenStateData = SerializerUtility.Serialize(fullscreenState);
                File.WriteAllText(Path.Combine(projectLibraryPath, FullscreenStateFilename), fullscreenStateData);
            }
            catch (IOException e)
            {
                if (EWFDebugging.Enabled)
                    Debug.LogException(e);
            }
        }
        internal static void LoadFullscreenState()
        {
            var winFocusedBeforeLoadState = EditorWindow.focusedWindow;
            try
            {
                string fullscreenStateData = File.ReadAllText(Path.Combine(projectLibraryPath, FullscreenStateFilename));
                fullscreenState = SerializerUtility.Deserialize<FullscreenState>(fullscreenStateData);
            }
            catch (FileNotFoundException)
            {
            }
            catch (System.Exception e)
            {
                Debug.LogException(e);
            }

            if (fullscreenState.window == null)
                fullscreenState.window = new List<WindowFullscreenState>();

            var allFullscreenStates = fullscreenState.window.ToArray();
            WindowFullscreenState mainWindowFullscreenState = null;

            //Load types from assembly qualified names
            foreach (var state in allFullscreenStates)
            {
                try
                {
                    state.ActualType = Type.GetType(state.actualTypeAssemblyQualifiedName);
                    state.WindowType = Type.GetType(state.windowTypeAssemblyQualifiedName);
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

            //Re-assign recreated window instances to their fullscreen states
            var allWins = Resources.FindObjectsOfTypeAll<EditorWindow>();
            var unassignedFullscreenWins = new List<EditorWindow>();

            foreach (var win in allWins)
            {
                unassignedFullscreenWins.Add(win);
            }
            foreach (var state in allFullscreenStates)
            {
                if (state.EditorWin != null)
                {
                    unassignedFullscreenWins.Remove(state.EditorWin);
                }
                else if (state.WindowType == MainWindowType)
                {
                    mainWindowFullscreenState = state;
                }
                else if (state.IsFullscreen)
                {
                    bool foundMatch = false;

                    //Find with identifier
                    foreach (var win in unassignedFullscreenWins)
                    {
                        if (win.GetType() == state.ActualType)
                        {
                            var container = win.GetContainerWindow();
                            var containerTitle = win.GetContainerTitle();
                            float serializedID = 0f;
                            if (container != null)
                            {
                                var containerMaxSizeFI = ContainerWindowType.GetField("m_MaxSize", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                                var containerMaxSize = Vector2.zero;
                                if (containerMaxSizeFI != null) containerMaxSize = (Vector2)containerMaxSizeFI.GetValue(container);
                                serializedID = containerMaxSize.x;
                            }

                            if ((containerTitle != null && containerTitle == state.identifierTitle) || serializedID > 1f && Mathf.Abs(serializedID - state.serializedID) < 0.005f)
                            {
                                state.SetEditorWin(win);
                                foundMatch = true;
                                unassignedFullscreenWins.Remove(win);
                                break;
                            }
                        }
                    }
                    if (!foundMatch)
                    {
                        //Otherwise, find with xy match.
                        foreach (var win in unassignedFullscreenWins)
                        {
                            if (win.GetType() == state.ActualType)
                            {
                                var containerPosition = win.GetContainerPosition();
#if UNITY_EDITOR_OSX == false
                                bool xyMatches = (containerPosition.x == state.ContainerPosition.x && containerPosition.y == state.ContainerPosition.y);
#else
                                bool xyMatches = Mathf.Abs(containerPosition.height - state.ContainerPosition.height) < 200 && state.ContainerPosition.Contains(containerPosition.center) && Math.Abs(state.ContainerPosition.width - containerPosition.width) <= 1;
#endif
                                if (xyMatches)
                                {
                                    state.SetEditorWin(win);
                                    unassignedFullscreenWins.Remove(win);
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            loadedInitialState = true;

            //Find the window which was focused
            var focusedWindow = fullscreenState.window.Find(state => state.HasFocus == true);
            var setFullscreenForStates = new List<WindowFullscreenState>();

            //Decide which fullscreen windows to remake
            for (int i = 0; i < allFullscreenStates.Length; i++)
            {
                var state = allFullscreenStates[i];
                if (state != null && state.IsFullscreen)
                {
                    state.fullscreenPositionWasSet = false; //Makes sure the position is set again.

                    if (state.EditorWin != null)
                    {
                        state.currentlyRestoringFromState = true;
                        setFullscreenForStates.Add(state);
                    }
                    else if (state.WindowType != MainWindowType)
                    {
                        //Make sure the state is still in the dictionary. It could have been removed in a previous ToggleFullscreen call (in CleanDeletedWindows, because it didn't have an EditorWin yet).
                        if (!fullscreenState.window.Contains(state)) fullscreenState.window.Add(state);
                        state.currentlyRestoringFromState = true;
                        setFullscreenForStates.Add(state);
                    }
                }
            }

            if (mainWindowFullscreenState != null && mainWindowFullscreenState.IsFullscreen)
            //Set main window fullscreen state
            {
                mainWindowFullscreenState.currentlyRestoringFromState = true;
#if UNITY_EDITOR_OSX
                EditorMainWindow.position = mainWindowFullscreenState.PreFullscreenPosition;
                EditorMainWindow.SetFullscreen(true, mainWindowFullscreenState.ShowTopToolbar, mainWindowFullscreenState.FullscreenAtPosition, true);
           
#else
                if (FirstLoadSinceStartup)
                {
                    //Close the main window fullscreen state if we just started up and it was fullscreen on Unity close
                    EditorMainWindow.SetFullscreen(false, mainWindowFullscreenState.ShowTopToolbar, mainWindowFullscreenState.FullscreenAtPosition, true);
                }
                else
                {
                    if (mainWindowFullscreenState.containerWindow == null || mainWindowFullscreenState.originalContainerWindow == null)
                    {
                        fullscreenState.window.Remove(mainWindowFullscreenState); //Remove the old fullscreen state because the originalContainer needs to be reset
                    }
                    EditorMainWindow.SetFullscreen(true, mainWindowFullscreenState.ShowTopToolbar, mainWindowFullscreenState.FullscreenAtPosition, true);
                }
#endif
            }

            RunAfter(() => true, () =>
            { //Run on the first update
                //Recreate other fullscreens after the main window
                for (int i = 0; i < setFullscreenForStates.Count; i++)
                {
                    var state = setFullscreenForStates[i];

                    if (state.EditorWin != null)
                    {
                        state.EditorWin.SetFullscreen(true, state.FullscreenAtPosition, state.ShowTopToolbar);
                    }
                    else
                    {
                        //Window doesn't exist any more - it may have been closed manually.
                    }
                }

                //Remove fullscreen windows which don't have a fullscreen state
                int unassignedWinsClosed = 0;
                ////Find any remaining fullscreen windows (which could have ended up on another screen) and assign to unassigned states if any more exist.
                foreach (var win in unassignedFullscreenWins)
                {
                    if (win == null) continue;
#if UNITY_EDITOR_WIN
                    bool isFSOnDisplay = win.GetShowMode() == EditorWindowExtensions.ShowMode.PopupMenu;
#else
                    bool isFSOnDisplay = win.IsFullscreenOnDisplay(EditorDisplay.ClosestToPoint(win.GetContainerPosition().center));
#endif
                    bool dockedWithAnotherFSstate = setFullscreenForStates.Find(state => state.EditorWin != null && object.Equals(state.EditorWin.GetContainerWindow(), win.GetContainerWindow())) != null;

                    if (isFSOnDisplay && !dockedWithAnotherFSstate)
                    {
                        if (win.GetContainerWindow() != null)
                        {
                            win.Close();
                        }
                        else
                        {
                            UnityEngine.Object.DestroyImmediate(win, true);
                        }
                        unassignedWinsClosed++;
                    }
                }
                if (unassignedWinsClosed > 0) EWFDebugging.LogWarning("Couldn't find state for " + unassignedWinsClosed + " fullscreen windows. Closed them.", true);

                fullscreenState.CleanDeletedWindows();

                //Refocus the main window
                EditorMainWindow.Focus();

                //Bring any fullscreen window which is on top of the main window to the front.
                try
                {
                    var windowOverMain = fullscreenState.window.Find(state => state.IsFullscreen && state.EditorWin != null && EditorDisplay.ClosestToPoint(state.FullscreenAtPosition).Bounds == EditorDisplay.ClosestToPoint(EditorMainWindow.position.center).Bounds);
                    if (windowOverMain != null)
                    {
                        GiveFocusAndBringToFront(windowOverMain.EditorWin);
                    }
                }
                catch { }

                //Refocus the window which was previously focused
                if (focusedWindow != null && focusedWindow.EditorWin != null)
                {
                    GiveFocusAndBringToFront(focusedWindow.EditorWin);
                }
                else
                {
                    GiveFocusAndBringToFront(winFocusedBeforeLoadState);
                }

                //Toggle fullscreen for states which were queued up before load was complete
                for (int i = 0; i < queuedStatesToToggleOnLoad.Count; i++)
                {
                    var state = queuedStatesToToggleOnLoad[i];
                    if (state != null)
                        ToggleFullscreen(state, state.CloseOnExitFullscreen, state.FullscreenAtPosition, state.ShowTopToolbar, state.CreatedAtGameStart);
                }
                queuedStatesToToggleOnLoad.Clear();
                if (RunOnNextLoadMethods != null)
                {
                    RunOnNextLoadMethods.Invoke();
                    RunOnNextLoadMethods = null;
                }
            }, 10, true);
        }

        internal static bool RunAfterInitialStateLoaded(RunOnLoad methodToRun)
        {
            if (!loadedInitialState)
            {
                RunOnNextLoadMethods -= methodToRun;
                RunOnNextLoadMethods += methodToRun;
                return true;
            }
            return false;
        }

        internal static void GiveFocusAndBringToFront(EditorWindow focusWin)
        {
            if (focusWin == null) return;
            var focused = EditorWindow.GetWindow(focusWin.GetType(), false, focusWin.GetWindowTitle(), true);
            if (focused != focusWin && focused.GetWindowTitle() == focusWin.GetWindowTitle())
                focused.Close();
            focusWin.Focus();
        }

        /// <summary>
        /// Get the EditorWindow which currently has the mouse over it.
        /// </summary>
        public static EditorWindow GetMouseOverWindow()
        {
            EditorWindow mouseOverWin = null;
            if (EditorWindow.mouseOverWindow != null) mouseOverWin = EditorWindow.mouseOverWindow;
            else if (EditorWindow.focusedWindow != null && EditorWindow.focusedWindow.position.Contains(EditorInput.MousePosition))
                mouseOverWin = EditorWindow.focusedWindow;
            else if (EditorFullscreenState.fullscreenState.window != null)
            {
                var allWinStates = EditorFullscreenState.fullscreenState.window.ToArray();
                foreach (var win in allWinStates)
                {
                    if (win.EditorWin != null && win.EditorWin.position.Contains(EditorInput.MousePosition))
                    {
                        mouseOverWin = win.EditorWin;
                        break;
                    }
                }
            }

            return mouseOverWin;
        }

        /// <summary>
        /// Returns the fullscreen opening position of the specified window or window type according to the current options.
        /// </summary>
        internal static Vector2 GetOptionsSpecifiedFullscreenOpenAtPosition(EditorWindow editorWin, Type windowType, EditorFullscreenSettings.FullscreenOption fullscreenOptions)
        {
            var openAtPosition = fullscreenOptions.openAtPosition;
            Vector2 atPosition = Vector2.zero;

            switch (openAtPosition)
            {
                case EditorFullscreenSettings.OpenFullscreenAtPosition.AtCurrentWindowPosition:
                    if (editorWin != null)
                    {
                        atPosition = editorWin.GetPointOnWindow();
                    }
                    else
                    {
                        var wins = Resources.FindObjectsOfTypeAll<EditorWindow>();
                        bool foundWin = false;
                        foreach (var win in wins)
                        {
                            if (GetWindowType(windowType) == win.GetWindowType())
                            {
                                atPosition = win.GetContainerPosition().center;
                                foundWin = true;
                                break;
                            }
                        }
                        if (!foundWin)
                            atPosition = EditorMainWindow.position.center;
                    }
                    break;
                case EditorFullscreenSettings.OpenFullscreenAtPosition.AtMousePosition:
                    atPosition = EditorInput.MousePosition;
                    break;
                case EditorFullscreenSettings.OpenFullscreenAtPosition.AtSpecifiedPosition:
                    atPosition = fullscreenOptions.position;
                    break;
                case EditorFullscreenSettings.OpenFullscreenAtPosition.AtSpecifiedPositionAndSize:
                    atPosition = fullscreenOptions.position;
                    break;
                case EditorFullscreenSettings.OpenFullscreenAtPosition.None:
                    break;
            }
            return atPosition;
        }

        /// <summary>
        /// Returns true if a window type is fullscreen on the screen specified by the given options.
        /// </summary>
        public static bool WindowTypeIsFullscreenAtOptionsSpecifiedPosition(Type windowType, EditorFullscreenSettings.FullscreenOption fullscreenOptions)
        {
            if (windowType == MainWindowType) return EditorMainWindow.IsFullscreenAtPosition(GetOptionsSpecifiedFullscreenOpenAtPosition(null, windowType, fullscreenOptions));
            var openAtPosition = fullscreenOptions.openAtPosition;
            bool isFullscreen = false;

            switch (openAtPosition)
            {
                case EditorFullscreenSettings.OpenFullscreenAtPosition.AtMousePosition:
                    EditorWindow mouseOverWin = EditorFullscreenState.GetMouseOverWindow();
                    if (mouseOverWin != null && mouseOverWin.GetWindowType() == windowType)
                    {
                        isFullscreen = mouseOverWin.IsFullscreen();
                    }
                    else
                    {
                        if (fullscreenState.window == null) break;
                        foreach (var state in fullscreenState.window)
                        {
                            if (state.IsFullscreen && state.EditorWin != null && state.WindowType == windowType && state.EditorWin.IsFullscreenOnDisplay(EditorDisplay.ClosestToPoint(EditorInput.MousePosition)))
                            {
                                isFullscreen = true;
                                break;
                            }
                        }
                    }
                    break;
                case EditorFullscreenSettings.OpenFullscreenAtPosition.None:
                    isFullscreen = false;
                    break;
                default:
                    Vector2 openAtPos = GetOptionsSpecifiedFullscreenOpenAtPosition(null, windowType, fullscreenOptions);
                    isFullscreen = WindowTypeIsFullscreenOnScreenAtPosition(windowType, openAtPos);
                    break;
            }
            return isFullscreen;
        }

        /// <summary>
        /// Returns true if a window type is fullscreen on the screen at the specified position.
        /// </summary>
        public static bool WindowTypeIsFullscreenOnScreenAtPosition(Type windowType, Vector2 atPosition)
        {
            bool isFullscreen = false;

            //Main Window
            if (windowType == MainWindowType) return EditorMainWindow.IsFullscreenAtPosition(atPosition);

            var wins = Resources.FindObjectsOfTypeAll<EditorWindow>();
            foreach (var editorWin in wins)
            {
                if (editorWin.GetType() == windowType || editorWin.GetWindowType() == windowType)
                {
                    var winState = FindWindowState(editorWin);
                    if (winState != null && winState.IsFullscreen)
                    {
                        var atPositionScreenCenter = EditorDisplay.ClosestToPoint(atPosition).Bounds.center; //Position could be outside of any screen, so use the center of the screen we're actually fullscreening on.
                        if (editorWin.GetContainerPosition().Contains(atPositionScreenCenter) && editorWin.IsFullscreen())
                        {
                            isFullscreen = true;
                            break;
                        }
                    }
                }
            }
            return isFullscreen;
        }

        /// <summary>
        /// Toggle fullscreen at a position decided according to the current options.
        /// </summary>
        public static bool ToggleFullscreenUsingOptions(Type windowType, EditorFullscreenSettings.FullscreenOption fullscreenOps)
        {
            return ToggleFullscreenUsingOptions(null, windowType, fullscreenOps, false, false);
        }

        /// <summary>
        /// Toggle fullscreen according to some fullscreen options.
        /// </summary>
        public static bool ToggleFullscreenUsingOptions(EditorWindow editorWindow, Type windowType, EditorFullscreenSettings.FullscreenOption fullscreenOps, bool triggeredOnPlayStateChange, bool showTopTabs)
        {
            bool setFullscreen;
            ToggleFullscreenUsingOptions(editorWindow, windowType, fullscreenOps, triggeredOnPlayStateChange, showTopTabs, out setFullscreen);
            return setFullscreen;
        }

        /// <summary>
        /// Toggle fullscreen according to some fullscreen options, and return the window fullscreen state.
        /// </summary>
        public static WindowFullscreenState ToggleFullscreenUsingOptions(EditorWindow editorWindow, Type windowType, EditorFullscreenSettings.FullscreenOption fullscreenOps, bool triggeredOnPlayStateChange, bool showTopTabs, out bool setFullscreen)
        {
            var openAtPosition = fullscreenOps.openAtPosition;
            Vector2 atPosition = GetOptionsSpecifiedFullscreenOpenAtPosition(null, windowType, fullscreenOps);
            EWFDebugging.StartTimer("ToggleFullscreenAtOptionsSpecifiedPosition");
            if (windowType == null)
            {
                if (fullscreenOps != null && fullscreenOps.WindowType != null)
                    windowType = fullscreenOps.WindowType;
                else
                    EWFDebugging.LogError("Window type is null.");
            }

            WindowFullscreenState windowState = null;

            if (windowState == null) windowState = EditorFullscreenState.FindWindowState(editorWindow, windowType, EditorDisplay.ClosestToPoint(atPosition));
            windowState.ShowTopTabs = showTopTabs;
            windowState.fullscreenOptionID = fullscreenOps.OptionID; //All fullscreen states which are opened according to fullscreen options are assigned the FullscreenOptionID.

            if (openAtPosition == EditorFullscreenSettings.OpenFullscreenAtPosition.AtMousePosition)
            {
                setFullscreen = ToggleFullscreenAtMousePosition(windowState, windowType, fullscreenOps.showToolbarByDefault, triggeredOnPlayStateChange);
            }
            else
            {
                setFullscreen = ToggleFullscreen(windowState, atPosition, fullscreenOps.showToolbarByDefault, triggeredOnPlayStateChange);
            }
            EWFDebugging.LogTime("ToggleFullscreenAtOptionsSpecifiedPosition");
            return windowState;
        }

        /// <summary>
        /// Toggle fullscreen at the current mouse position for the window with the specified type.
        /// </summary>
        public static bool ToggleFullscreenAtMousePosition(Type windowType)
        {
            return ToggleFullscreenAtMousePosition(windowType, true, false);
        }

        /// <summary>
        /// Toggle fullscreen at the current mouse position for the window with the specified type.
        /// </summary>
        public static bool ToggleFullscreenAtMousePosition(Type windowType, bool showTopToolbar)
        {
            return ToggleFullscreenAtMousePosition(windowType, showTopToolbar, false);
        }

        /// <summary>
        /// Toggle fullscreen at the current mouse position for the window with the specified type.
        /// </summary>
        public static bool ToggleFullscreenAtMousePosition(Type windowType, bool showTopToolbar, bool triggeredOnPlayStateChange)
        {
            return ToggleFullscreenAtMousePosition(null, windowType, showTopToolbar, triggeredOnPlayStateChange);
        }

        /// <summary>
        /// Toggle fullscreen at the current mouse position for the window with the specified type.
        /// If no window state or type are given, fullscreens the current MouseOverWindow.
        /// </summary>
        private static bool ToggleFullscreenAtMousePosition(WindowFullscreenState windowState, Type windowType, bool showTopToolbar, bool triggeredOnPlayStateChange)
        {
            Vector2 mousePos = EditorInput.MousePosition;
            if (windowState == null && windowType == null)
            {
                EWFDebugging.LogError("Window state and type cannot both be null.");
                return false;
            }
            else if (windowState == null)
                windowState = EditorFullscreenState.FindWindowState(null, windowType, EditorDisplay.ClosestToPoint(mousePos));
            else if (windowType == null) windowType = windowState.WindowType;

            if (windowState.WindowType != null)
            {
                return EditorFullscreenState.ToggleFullscreen(windowState, mousePos, showTopToolbar, triggeredOnPlayStateChange);
            }
            return false;
        }

        /// <summary>
        /// Toggle fullscreen for a window type (Creates a new fullscreen window if none already exists).
        /// </summary>
        /// <param name="windowType">The type of the window to create a fullscreen for.</param>
        /// <returns>True if the window type became fullscreen. False if fullscreen was exited.</returns>
        public static bool ToggleFullscreen(Type windowType)
        {
            return ToggleFullscreen(FindWindowState(windowType), true);
        }

        /// <summary>
        /// Toggle fullscreen for a window type, on the screen at a position.
        /// </summary>
        /// <param name="windowType">The type of the window to create a fullscreen for.</param>
        /// <param name="atPosition">Fullscreen will be toggled on the screen which is at this position.</param>
        /// <returns>True if the window type became fullscreen. False if fullscreen was exited.</returns>
        public static bool ToggleFullscreen(Type windowType, Vector2 atPosition)
        {
            return ToggleFullscreen(windowType, atPosition, true, false);

        }

        /// <summary>
        /// Toggle fullscreen for a window type, on the screen at a position.
        /// </summary>
        /// <param name="windowType">The type of the window to create a fullscreen for.</param>
        /// <param name="atPosition">Fullscreen will be toggled on the screen which is at this position.</param>
        /// <param name="showTopToolbar">Show the top toolbar by default if opening a fullscreen.</param>
        /// <param name="showTopTabs">Show the top tabs in the fullscreen window (the enclosing docking area becomes fullscreen instead of the window itself).</param>
        /// <returns>True if the window type became fullscreen. False if fullscreen was exited.</returns>
        public static bool ToggleFullscreen(Type windowType, Vector2 atPosition, bool showTopToolbar, bool showTopTabs)
        {
            var windowState = EditorFullscreenState.FindWindowState(null, windowType, EditorDisplay.ClosestToPoint(atPosition));
            windowState.ShowTopTabs = showTopTabs;
            return ToggleFullscreen(windowState, atPosition, showTopToolbar, false);
        }

        /// <summary> Toggle fullscreen for a window type, on the screen at a position </summary>
        private static bool ToggleFullscreen(WindowFullscreenState windowState, Vector2 atPosition, bool showTopToolbar, bool triggeredOnPlayStateChange)
        {
            bool createNewWindow = (windowState.EditorWin == null && windowState.containerWindow == null);
            if (!createNewWindow && windowState.ShowTopTabs && windowState.EditorWin != null && windowState.EditorWin.IsInMainWin())
            {
                //Create a new window anyway if the existing window is docked to the main window. 
                createNewWindow = true;
            }
            return ToggleFullscreen(windowState, createNewWindow, atPosition, showTopToolbar, triggeredOnPlayStateChange);
        }

        /// <summary> Toggle fullscreen for a window type </summary>
        private static bool ToggleFullscreen(WindowFullscreenState windowState, bool createNewWindow, Vector2 atPosition, bool showTopToolbar, bool triggeredOnPlayStateChange)
        {
            if (showTopToolbar) windowState.ShowTopToolbar = true;
            if (triggeredOnPlayStateChange && !windowState.IsFullscreen) windowState.CreatedAtGameStart = true;
            return ToggleFullscreen(windowState, createNewWindow, atPosition);
        }

        /// <summary> Toggle fullscreen for a window state </summary>
        private static bool ToggleFullscreen(WindowFullscreenState windowState, bool createNewWindow)
        {
            if (windowState.EditorWin != null && windowState.IsFullscreen)
            {
                return ToggleFullscreen(windowState, createNewWindow, windowState.EditorWin.GetFullscreenDisplay().Bounds.center);
            }
            else if (windowState.WindowType == MainWindowType)
            {
                return ToggleFullscreen(windowState, createNewWindow, EditorMainWindow.position.center);
            }
            else
            {
                return ToggleFullscreen(windowState, createNewWindow, EditorDisplay.PrimaryDesktopResolution.center);
            }
        }

        /// <summary> Toggle fullscreen for a window state, on the screen at position </summary>
        private static bool ToggleFullscreen(WindowFullscreenState windowState, bool createNewWindow, Vector2 atPosition)
        {
            if (windowState.EditorWin != null && windowState.EditorWin.position.width < 10 || windowState.WindowType == typeof(EditorInput)) return false;
            var winType = windowState.WindowType.ToString();

            if (winType.Contains("FsmEditor") //To support PlayMaker window, only keep 1 instance
             || winType.Contains("Timeline")) //To support Timeline window, only keep 1 instance
            {
                createNewWindow = false;
                windowState.ShowTopTabs = true;
            }

#if UNITY_EDITOR_WIN && UNITY_2019_3_OR_NEWER
            //windowState.ShowTopTabs = true; //Always show top tabs from 2019.3 (Otherwise top toolbar doesn't hide correctly).
#endif
#if UNITY_EDITOR_OSX
            var editorDisplay = EditorDisplay.ClosestToPoint(atPosition);
            if (editorDisplay.Locked)
            {
                RunAfterDisplayNotLocked(atPosition, () => ToggleFullscreen(windowState, createNewWindow, atPosition));
                return (windowState != null) && !windowState.fullscreenPositionWasSet;
            }
#endif
            bool setFullscreen = false;
            if (windowState == null) throw new NullReferenceException("windowState is null. Cannot continue.");
            if (!createNewWindow && windowState.EditorWin == null && windowState.WindowType != MainWindowType)
            {
                //Not creating a new window, and not given a window, so decide which existing window to fullscreen.
                if (EditorWindow.mouseOverWindow != null && EditorWindow.mouseOverWindow.GetType() == windowState.WindowType) windowState.SetEditorWin(EditorWindow.mouseOverWindow);
                else if (EditorWindow.focusedWindow != null && EditorWindow.focusedWindow.GetType() == windowState.WindowType) windowState.SetEditorWin(EditorWindow.focusedWindow);
                else
                {
                    EditorWindow[] allWinsOfType = null;
                    try
                    {
                        allWinsOfType = (EditorWindow[])Resources.FindObjectsOfTypeAll(windowState.WindowType);
                    }
                    catch (Exception e) { if (EWFDebugging.Enabled) EWFDebugging.LogError(e.ToString()); return false; }
                    foreach (var win in allWinsOfType)
                    {
                        if (!win.IsInMainWin()) windowState.SetEditorWin(win); //Choose standalone windows over ones docked in the main view.
                        break;
                    }
                    foreach (var win in allWinsOfType)
                    {
                        if (!win.IsFullscreen()) windowState.SetEditorWin(win);
                        break;
                    }
                    if (windowState.EditorWin == null)
                    {
                        //There are no existing windows to fullscreen
                    }
                }
            }

            if (!loadedInitialState)
            {
                windowState.CloseOnExitFullscreen = createNewWindow;
                windowState.FullscreenAtPosition = atPosition;
                queuedStatesToToggleOnLoad.Add(windowState);

                if (EWFDebugging.Enabled)
                    EWFDebugging.LogLine("The fullscreen state hasn't been loaded yet. Fullscreen toggle has been queued.");
                else
                    return false;
            }
            if (windowState.WindowType == MainWindowType)
            {
                setFullscreen = EditorMainWindow.ToggleFullscreen(windowState.ShowTopToolbar, atPosition);
            }
            else if (windowState.EditorWin == null || (!windowState.IsFullscreen && createNewWindow))
            {
                if (windowState.ActualType == typeof(SceneView)) { windowState.ActualType = typeof(CustomSceneView); } //Always create CustomSceneView for SceneView
                EditorWindow win = EditorWindowExtensions.CreateWindow(windowState.ActualType);
                windowState.CreatedNewWindow = true;

                //Clone state of existing window if the state already had one.
                if (windowState.EditorWin != null && windowState.EditorWin.GetType() == windowState.ActualType)
                {
                    var serializedWin = EditorJsonUtility.ToJson(windowState.EditorWin);
                    EditorJsonUtility.FromJsonOverwrite(serializedWin, win);
                }


                windowState.SetEditorWin(win);

                if (!windowState.ShowTopTabs && windowState.CloseOnExitFullscreen)
                {
                    if (windowState.ActualType == typeof(CustomSceneView) || win.GetWindowType() == GameViewType)
                        win.SetWindowTitle(windowState.WindowName, true); //Reset title content on custom Scene and Game views to prevent icon not found error.
                }

                windowState.CloseOnExitFullscreen = true; //Since we are creating a new window, this window should close when fullscreen is exited

                setFullscreen = win.ToggleFullscreen(atPosition);
            }
            else
            {
                //The window already exists.
                if (windowState.IsFullscreen) atPosition = windowState.FullscreenAtPosition; //If the window is already fullscreen it should be exited, so set the atPosition to the place it was fullscreened at.
                setFullscreen = windowState.EditorWin.ToggleFullscreen(atPosition);
            }

            return setFullscreen;
        }

        /// <summary> Toggle the toolbar in the active window if it is fullscreen </summary>
        public static bool ToggleToolbarInFullscreen()
        {
            EWFDebugging.LogLine("Toggling Top Toolbar");
            bool setShowTopToolbar = false;
            var fullscreenWinState = FocusedWindowState;

            //If no fullscreen window is focused, toggle the top tab for the last window state to change status (enter fullscreen or toggle top tabs).
            if ((fullscreenWinState == null || !fullscreenWinState.IsFullscreen) && lastWinStateToChangeFullscreenStatus != null)
            {
                if (lastWinStateToChangeFullscreenStatus.EditorWin != null || lastWinStateToChangeFullscreenStatus.WindowType == MainWindowType)
                    fullscreenWinState = lastWinStateToChangeFullscreenStatus;
            }

            if (fullscreenWinState == null || !fullscreenWinState.IsFullscreen)
            {
                //If the current EditorWindow state isn't fullscreen, toggle the toolbar of the main window if that is fullscreen.
                fullscreenWinState = EditorMainWindow.GetWindowFullscreenState();
            }

            if (fullscreenWinState != null && fullscreenWinState.IsFullscreen)
            {
                lastWinStateToChangeFullscreenStatus = fullscreenWinState;
                setShowTopToolbar = !fullscreenWinState.ShowTopToolbar;
                bool toggled = ToggleToolbarInFullscreen(fullscreenWinState, setShowTopToolbar);

                return toggled;
            }
            else
            {
                return false;
            }
        }
        /// <summary> Toggle the toolbar for the specified window state, if it is fullscreen </summary>
        public static bool ToggleToolbarInFullscreen(WindowFullscreenState fullscreenWinState, bool showTopToolbar)
        {
            if (fullscreenWinState != null && fullscreenWinState.IsFullscreen)
            {
                if (fullscreenWinState.WindowType == MainWindowType)
                {
                    EditorMainWindow.SetFullscreen(true, showTopToolbar);
                }
                else if (fullscreenWinState.EditorWin != null)
                {
                    fullscreenWinState.EditorWin.SetFullscreen(true, showTopToolbar);

                }
            }
            return showTopToolbar;
        }

        public static Type GetWindowType(Type derivedType)
        {
            Type windowType = null;
            try
            {
                MethodInfo getWindowType = derivedType.GetMethod("GetWindowType", BindingFlags.Public | BindingFlags.Static);
                windowType = (Type)getWindowType.Invoke(null, null);
            }
            catch
            {
                Type baseType = derivedType.BaseType;
                if (baseType != null && (baseType == typeof(SceneView) || baseType == GameViewType))
                {
                    windowType = baseType;
                }
            }

            if (windowType != null)
                return windowType;
            else
                return derivedType;
        }

        internal static void LongToFloats(long num, out float leftBits, out float rightBits)
        {
            var bits = BitConverter.GetBytes(num);
            leftBits = BitConverter.ToSingle(bits, 0);
            rightBits = BitConverter.ToSingle(bits, 4);
        }

        internal static long LongFromFloats(float leftBits, float rightBits)
        {
            var bits = new List<byte>(64);
            bits.InsertRange(0, BitConverter.GetBytes(leftBits));
            bits.InsertRange(4, BitConverter.GetBytes(rightBits));
            return BitConverter.ToInt64(bits.ToArray(), 0);
        }

        internal static void TriggerFullscreenEvent(object window, Type windowType, Vector2 atPosition, bool enteredFullscreen)
        {
            if (enteredFullscreen)
            {
                EditorFullscreenSettings.settings.fullscreenedCount++;
                EditorFullscreenSettings.settings.SaveSettings(false);
            }
            if (FullscreenEventHandler != null)
            {
                FullscreenEventHandler.Invoke(window, windowType, atPosition, enteredFullscreen);
            }
        }

        internal delegate bool ConditionToMeet();
        internal static void RunAfter(ConditionToMeet conditionFunction, EditorApplication.CallbackFunction callback, int maxCount, bool runIfMaxCountReached)
        {
            EditorApplication.CallbackFunction runAfterConditionMet = null;
            int setCount = 0;
            runAfterConditionMet = () =>
            {
                setCount++;
                if (setCount >= maxCount || conditionFunction.Invoke())
                {
                    EditorApplication.update -= runAfterConditionMet;
                    bool runCallback = runIfMaxCountReached || setCount < maxCount;
                    setCount = maxCount + 1;
                    try
                    {
                        if (runCallback) callback.Invoke();
                    }
                    catch (Exception e)
                    {
                        EWFDebugging.LogError(e.ToString(), true);
                    }

                }
            };
            EditorApplication.update += runAfterConditionMet;
        }

        internal static void RunAfterDisplayNotLocked(Vector2 displayAtPoint, EditorApplication.CallbackFunction callback)
        {
            var editorDisplay = EditorDisplay.ClosestToPoint(displayAtPoint);
            EditorApplication.CallbackFunction displayNoLongerLocked = null;
            int maxCount = 150;
            int setCount = 0;
            displayNoLongerLocked = () =>
            {
                setCount++;
                if (setCount > maxCount)
                {
                    EditorApplication.update -= displayNoLongerLocked;
                }
                else if (setCount == maxCount || !editorDisplay.Locked)
                {
                    EditorApplication.update -= displayNoLongerLocked;
                    setCount = 1000;
                    editorDisplay.Locked = false; //Release the lock anyway if the max count was reached.
                    try
                    {
                        callback.Invoke();
                    }
                    catch (Exception e)
                    {
                        EWFDebugging.LogError(e.ToString() + "\n\n" + ((e.InnerException != null) ? e.InnerException.ToString() : ""), true);
                    }
                }
            };
            EditorApplication.update += displayNoLongerLocked;
        }
    }
}