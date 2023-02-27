/* 
 * Author:  Johanan Round
 * Package: Editor Window Fullscreen
 * License: Unity Asset Store EULA (Editor extension asset. Requires 1 license per machine.)
 */

using UnityEngine;
using UnityEditor;
using System;
using System.IO;
using System.Reflection;
using System.Linq;
using System.Collections.Generic;

namespace EditorWindowFullscreen
{
    public sealed class EditorFullscreenSettings
    {
        public const int EWF_MAJOR_VERSION = 1;
        public const int EWF_MINOR_VERSION = 3;
        public const int EWF_PATCH_VERSION = 1;
        public const int SETTINGS_VERSION_NUMBER = 131;
        public const string MENU_ITEM_PATH = "Window/Editor Window Fullscreen/";
        public static string SettingsSaveVersion { get { return SETTINGS_VERSION_NUMBER + "_" + Application.unityVersion; } }
        public static string Version { get { return EditorFullscreenSettings.EWF_MAJOR_VERSION + "." + EditorFullscreenSettings.EWF_MINOR_VERSION + "." + EditorFullscreenSettings.EWF_PATCH_VERSION; } }
        private static EditorFullscreenSettings _settings;
        public static EditorFullscreenSettings settings
        {
            get
            {
                if (_settings == null)
                {
                    Initialize();
                }
                return _settings;
            }
        }

        public bool debugModeEnabled = false;
        public bool fullscreenNotification = true;
#if UNITY_EDITOR_OSX
        public float fullscreenNotificationDuration = 1.5f; //Min 0, max 10
#else
        public float fullscreenNotificationDuration = 1; //Min 0, max 10
#endif

        [SerializeField]
        private int nextFullscreenOptionID = 1;
        public int fullscreenedCount = 0;
        public int clickedLeaveRevCount = 0;
        public int settingsClosedAfterRevCount = 0;
        private Dictionary<int, FullscreenOption> fullscreenOptions = new Dictionary<int, FullscreenOption>();
        private bool customWindowFullscreenOptionsPopulated = false;
        private void PopulateCustomWindowFullscreenOptions()
        {
            //Populate fullscreenOptions dictionary for custom windows.
            if (this != null && customWindows != null)
            {
                foreach (var customWin in customWindows)
                {
                    if (!fullscreenOptions.ContainsKey(customWin.OptionID))
                    {
                        fullscreenOptions.Add(customWin.OptionID, customWin);
                    }
                }
            }
        }

        public FullscreenOption GetFullscreenOption(int optionID)
        {
            FullscreenOption fullscreenOps = null;
            if (!customWindowFullscreenOptionsPopulated)
            {
                PopulateCustomWindowFullscreenOptions();
                customWindowFullscreenOptionsPopulated = true;
            }
            fullscreenOptions.TryGetValue(optionID, out fullscreenOps);
            return fullscreenOps;
        }

        public enum OpenFullscreenAtPosition
        {
            None = 0,
            AtCurrentWindowPosition = 1,
            AtMousePosition = 2,
            AtSpecifiedPosition = 3,
            AtSpecifiedPositionAndSize = 4
        }

        public enum CloseFullscreenOnGameStop
        {
            None = 0,
            FullscreensCreatedAtGameStart = 1,
            AllFullscreenGameWindows = 2
        }

        public enum StopGameWhenExitingFullscreen
        {
            Never = 0,
            WhenAnyFullscreenGameViewIsExited = 1,
            WhenAllFullscreenGameViewsExited = 2
        }

        public enum ImproveFPSOptions
        {
            DoNothing = 0,
            HideDockedGameView = 1,
            CloseAllOtherGameWindows = 2
        }

        //Fullscreen Window Hotkey Defaults
        public FullscreenOption mainUnityWindow;
        public FullscreenOption sceneWindow;
        public FullscreenOption gameWindow;
        public FullscreenOption currentlyFocusedWindow;
        public FullscreenOption windowUnderCursor;

        //Other Hotkey Defaults
        public FullscreenOption toggleTopToolbar;
        public FullscreenOption closeAllFullscreenWindows;
        public FullscreenOption resetToDefaultLayout;

        //Custom Windows
        public List<FullscreenOption> customWindows;

        //All Custom Game Windows (Including primary fullscreen game window).
        public List<FullscreenOption> AllGameWindows
        {
            get
            {
                var allGameWins = new List<FullscreenOption>();
                allGameWins.Add(gameWindow);

                if (customWindows != null)
                    allGameWins.AddRange(customWindows.Where(w => w.isGameView));

                return allGameWins;
            }
        }

        public EditorFullscreenSettings()
        {
            //Fullscreen Window Hotkey Defaults
            mainUnityWindow = new FullscreenOption(this, KeyCode.F8, EventModifiers.None, OpenFullscreenAtPosition.AtMousePosition, true, Vector2.zero, "Main Unity Window", EditorFullscreenState.MainWindowType, true);
            sceneWindow = new FullscreenOption(this, KeyCode.F10, EventModifiers.None, OpenFullscreenAtPosition.AtMousePosition, true, Vector2.zero, "Scene Window", EditorFullscreenState.SceneViewType, true);
            gameWindow = new FullscreenOption(this, KeyCode.F11, EventModifiers.None, OpenFullscreenAtPosition.AtMousePosition, false, Vector2.zero, "Game Window", EditorFullscreenState.GameViewType, true, true);
            currentlyFocusedWindow = new FullscreenOption(this, KeyCode.F9, EventModifiers.None, OpenFullscreenAtPosition.AtMousePosition, true, Vector2.zero, "Currently Focused Window", null, true);
            windowUnderCursor = new FullscreenOption(this, KeyCode.F9, EventModifiers.Control, OpenFullscreenAtPosition.AtMousePosition, true, Vector2.zero, "Window Under Cursor", null, true);

            //Other Hotkey Defaults
            toggleTopToolbar = new FullscreenOption(this, KeyCode.F12, EventModifiers.None, OpenFullscreenAtPosition.None, false, Vector2.zero, "Show/Hide Top Toolbar", null, true);
            closeAllFullscreenWindows = new FullscreenOption(this, KeyCode.F8, EventModifiers.Control, OpenFullscreenAtPosition.None, false, Vector2.zero, "Close All Fullscreen Windows", null, true);

            resetToDefaultLayout = new FullscreenOption(this, KeyCode.F8, EventModifiers.Control | EventModifiers.Alt | EventModifiers.Shift, OpenFullscreenAtPosition.None, false, Vector2.zero, "Reset Unity to default layout", null, true);

#if UNITY_EDITOR_OSX
            //Fullscreen Window Hotkey Defaults (OSX)
            mainUnityWindow.modifiers = EventModifiers.Command;
            sceneWindow.modifiers = EventModifiers.Command;
            gameWindow.modifiers = EventModifiers.Command;
            currentlyFocusedWindow.modifiers = EventModifiers.Command;
            windowUnderCursor.modifiers = EventModifiers.Alt | EventModifiers.Command;
            //Other Hotkey Defaults (OSX)
            toggleTopToolbar.modifiers = EventModifiers.Command;
            closeAllFullscreenWindows.modifiers = EventModifiers.Alt | EventModifiers.Command;
            resetToDefaultLayout.modifiers = EventModifiers.Alt | EventModifiers.Shift | EventModifiers.Command;
#endif
        }




        //Game Window Options
        public bool startGameWhenEnteringFullscreen = false;
        public StopGameWhenExitingFullscreen stopGameWhenExitingFullscreen = StopGameWhenExitingFullscreen.Never;
        public CloseFullscreenOnGameStop closeFullscreenOnGameStop;
        public ImproveFPSOptions improveFpsOptions = ImproveFPSOptions.HideDockedGameView;

        private static string scriptFileSubPath = "";
        private static string scriptFilePath = "";
        private static bool initialized = false;

        private static void Initialize()
        {
            initialized = true;
            try
            {
                scriptFileSubPath = GetMenuItemScriptFileRelativePath();
                if (scriptFileSubPath == null) return;
                scriptFilePath = Application.dataPath + scriptFileSubPath;

                _settings = LoadSettings();

                if (MenuItemScriptNeedsRefresh())
                {
                    _settings.SaveSettings(false);
                    _settings.UpdateMenuItems();
                }
            }
            catch (System.Exception e)
            {
                if (EWFDebugging.Enabled)
                {
                    Debug.LogError("Settings failed to load.");
                    Debug.LogException(e);
                    EWFDebugging.LogError("Settings failed to load. " + e.Message);
                }
            }
        }

        public void ClearHotkeyConflicts(int optionID, KeyCode hotkey, EventModifiers modifiers)
        {
            if (hotkey == KeyCode.None) return;
            var fullscreenOptions = typeof(EditorFullscreenSettings).GetFields(BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Public).Where(field => (field.FieldType == typeof(FullscreenOption))).ToList();

            foreach (var fsOp in fullscreenOptions)
            {
                var fullscreenOption = (FullscreenOption)fsOp.GetValue(this);
                if (optionID != fullscreenOption.OptionID && EditorInput.HotkeysMatch(hotkey, modifiers, fullscreenOption.hotkey, fullscreenOption.modifiers))
                {
                    //Clear the conflicting hotkey
                    fullscreenOption.hotkey = KeyCode.None;
                    fullscreenOption.modifiers = EventModifiers.None;
                }
                fsOp.SetValue(this, fullscreenOption);
            }
        }

        public void SaveSettings()
        {
            SaveSettings(true);
        }

        public void SaveSettings(bool hotkeysWereChanged)
        {
            EditorPrefs.SetString("EditorFullscreenWindowSettings_VER", SettingsSaveVersion);
            EditorPrefs.SetString("EditorFullscreenWindowSettings_ALL", SerializerUtility.Serialize(this));
            if (hotkeysWereChanged) this.UpdateMenuItems();
        }

        public static void ReloadSettings()
        {
            _settings = LoadSettings();
        }

        public static EditorFullscreenSettings LoadSettings()
        {
            if (!initialized) Initialize(); //Make sure static settings are initialized.
            string settingsData = EditorPrefs.GetString("EditorFullscreenWindowSettings_ALL");
            EditorFullscreenSettings loadedSettings = null;

            if (!string.IsNullOrEmpty(settingsData))
                loadedSettings = SerializerUtility.Deserialize<EditorFullscreenSettings>(settingsData);

            if (loadedSettings == null) return new EditorFullscreenSettings();
            else
            {
                if (loadedSettings.gameWindow.gameViewOptions == null) loadedSettings.gameWindow.gameViewOptions = new GameViewOptions(); //For backwards compatibility.
                return loadedSettings;
            }
        }

        public static void ResetToDefaults()
        {
            _settings = new EditorFullscreenSettings();
        }

        private static string GetMenuItemScriptFolderPath()
        {
            var scriptableObj = ScriptableObject.CreateInstance(typeof(EditorTextEntryWindow));
            var scriptableObjScript = MonoScript.FromScriptableObject(scriptableObj);
            var scriptPath = AssetDatabase.GetAssetPath(scriptableObjScript);
            ScriptableObject.DestroyImmediate(scriptableObj);
            if (string.IsNullOrEmpty(scriptPath)) return null;
            int firstSlash = scriptPath.IndexOf("/");
            int lastSlash = scriptPath.LastIndexOf("/");
            int dirLength = lastSlash - firstSlash;
            scriptPath = scriptPath.Substring(firstSlash, dirLength);
            return scriptPath;
        }

        private static string GetMenuItemScriptFileRelativePath()
        {
            var scriptPath = GetMenuItemScriptFolderPath();
            if (string.IsNullOrEmpty(scriptPath)) return null;
            return scriptPath + "/EditorFullscreenControllerMenuItems.cs";
        }

        private static bool MenuItemScriptNeedsRefresh()
        {
            bool needsRefresh = false;
            if (EditorPrefs.GetString("EditorFullscreenWindowSettings_VER") != SettingsSaveVersion) return true;
            var latestGuid = EditorPrefs.GetString("EditorFullscreenWindowSettings_GUID");
            if (File.Exists(scriptFilePath))
            {
                string line = "";
                StreamReader sr = new StreamReader(scriptFilePath);
                if (sr.Peek() != -1) sr.ReadLine();
                if (sr.Peek() != -1) line = sr.ReadLine();

                int guidStart = line.IndexOf("{") + 1;
                int guidEnd = line.LastIndexOf("}") - 1;
                int guidLength = guidEnd - guidStart + 1;
                if (guidStart > 0 && guidLength > 0)
                {
                    needsRefresh = line.Substring(guidStart, guidLength) != latestGuid;
                }
                else
                {
                    needsRefresh = true;
                }
                sr.Close();
            }
            else needsRefresh = true;
            return needsRefresh;
        }

        private void UpdateMenuItems()
        {
            var settings = this;
            var guid = Guid.NewGuid().ToString();
            //Create all hotkeys as menu-items
            string script = "//AUTO-GENERATED SCRIPT. ANY MODIFICATIONS WILL BE OVERWRITTEN.\r\n"
                          + "//GUID: {" + guid + "}\r\n"
                          + "using UnityEngine;\r\nusing UnityEditor;\r\n"
                          + "namespace " + typeof(EditorFullscreenSettings).Namespace + " {\r\n    public class EditorFullscreenControllerMenuItems {\r\n";

            script += GetMenuItemScript("Toggle Main Window Fullscreen", 3, "ToggleMainWindowFullscreen", settings.mainUnityWindow, "mainUnityWindow", 21);
            script += GetMenuItemScript("Toggle Scene View Fullscreen", 4, "ToggleSceneViewFullscreen", settings.sceneWindow, "sceneWindow", 22);
            script += GetMenuItemScript("Toggle Game Fullscreen", 5, "ToggleGameViewFullscreen", settings.gameWindow, "gameWindow", 23);
            script += GetMenuItemScript("Toggle Focused Window Fullscreen", 2, "ToggleFocusedWindowFullscreen", settings.currentlyFocusedWindow, "currentlyFocusedWindow", 24);
            script += GetMenuItemScript("Toggle Mouseover Window Fullscreen", 2, "ToggleWindowUnderCursorFullscreen", settings.windowUnderCursor, "windowUnderCursor", 25);
            script += GetMenuItemScript("Show \u2215 Hide Toolbar in Fullscreen", 3, "ToggleTopToolbar", settings.toggleTopToolbar, "toggleTopToolbar", 26);
            script += GetMenuItemScript("Close all Editor Fullscreen Windows", 2, "CloseAllEditorFullscreenWindows", settings.closeAllFullscreenWindows, "closeAllFullscreenWindows", 27);
            script += GetMenuItemScript("Reset to Default Layout", 5, "ResetToDefaultLayout", settings.resetToDefaultLayout, "resetToDefaultLayout", 28);

            //Custom windows
            int gameWinNum = 1;
            if (settings.customWindows != null)
            {
                for (int i = 0; i < settings.customWindows.Count; i++)
                {
                    var optionID = settings.customWindows[i].OptionID;
                    if (settings.customWindows[i].isGameView) gameWinNum++;
                    string label = settings.customWindows[i].optionLabel;
                    if (label == "Game Window") label += " " + gameWinNum;
                    script += GetMenuItemScript(label, 2, "CustomWindow" + i, settings.customWindows[i], "GetFullscreenOption(" + optionID + ")", 50);
                }
            }

            script += "    }\r\n}\r\n";
            try
            {
                File.WriteAllText(scriptFilePath, script);
                EditorPrefs.SetString("EditorFullscreenWindowSettings_GUID", guid);
                string assetPath = ("Assets/" + scriptFileSubPath).Replace("//", "/"); //Replace double slashes to support some older versions which have an extra slash.
                AssetDatabase.ImportAsset(assetPath);
            }
            catch (IOException) { Debug.LogError("Write error. Could not write the menu items script to disk: " + scriptFilePath); }
        }

        private static string GetMenuItemScript(string label, int tabsInLabel, string methodName, EditorFullscreenSettings.FullscreenOption fullscreenOptions, string fullscreenHotkey, int menuPriority)
        {
            if (fullscreenOptions.hotkey == KeyCode.None) return ""; //Blank hotkey, so no script required.
            string hotkeyString = " " + EditorInput.GetKeyMenuItemString(fullscreenOptions.hotkey, fullscreenOptions.modifiers);

#if UNITY_EDITOR_OSX
            //Add extra label on macOS for F hotkeys (F hotkeys are labeled incorrectly with modifiers, and some other hotkeys show no label)
            for (int i=0; i < tabsInLabel; i++) {
                label += char.ConvertFromUtf32(9);
            }
            label += EditorInput.GetKeysDownString(fullscreenOptions.hotkey, fullscreenOptions.modifiers);
#endif

            string fullscreenOpsString = "EditorFullscreenSettings.settings." + fullscreenHotkey;
            string methodCall = "EditorFullscreenController.TriggerFullscreenHotkey(" + fullscreenOpsString + ");";

            string script = "        [MenuItem(" + typeof(EditorFullscreenSettings).Name + ".MENU_ITEM_PATH + \"" + label + hotkeyString + "\", false, " + menuPriority + ")]\r\n"
                          + "        public static void " + methodName + "() {" + methodCall + "}\r\n";
            return script;

        }

        private static string GetMenuItemScript(string label, string methodName, EditorFullscreenSettings.FullscreenOption fullscreenOptions)
        {
            string hotkeyString = EditorInput.GetKeyMenuItemString(fullscreenOptions.hotkey, fullscreenOptions.modifiers);
            string script = "        [MenuItem(" + typeof(EditorFullscreenSettings).Name + ".MENU_ITEM_PATH + \"" + label + " " + hotkeyString + "\")]\r\n"
                          + "        public static void " + methodName + "() {EditorFullscreenController." + methodName + "();}\r\n";
            return script;
        }

        public static string FormatCamelCaseName(string enumName)
        {
            enumName = System.Text.RegularExpressions.Regex.Replace(enumName, "(?!^)([A-Z])", " $1");
            return enumName;
        }

        [Serializable]
        public class FullscreenOption
        {
            public KeyCode hotkey;
            public EventModifiers modifiers;
            public OpenFullscreenAtPosition openAtPosition;
            public bool showToolbarByDefault;
            public Vector2 position;

            public string optionLabel;

            [SerializeField]
            private int optionID;
            /// <summary>Unique identifier for this option</summary>
            public int OptionID { get { return optionID; } }

            public string windowTypeAssemblyQualifiedName;
            public bool isGameView;
            public bool openOnGameStart;
            public GameViewOptions gameViewOptions;

            private void AssignOptionID(EditorFullscreenSettings settings, bool addToOptionsDict)
            {
                optionID = settings.nextFullscreenOptionID;
                settings.nextFullscreenOptionID++;
                if (addToOptionsDict) settings.fullscreenOptions[optionID] = this;
            }

            public FullscreenOption(EditorFullscreenSettings settings) { AssignOptionID(settings, false); }
            public FullscreenOption(EditorFullscreenSettings settings, bool addToOptionsDict) { AssignOptionID(settings, addToOptionsDict); }
            public FullscreenOption(EditorFullscreenSettings settings, KeyCode hotkey, EventModifiers modifiers, OpenFullscreenAtPosition openAtPosition, bool showToolbarByDefault, Vector2 position, string optionLabel, Type windowType, bool addToOptionsDict)
            : this(settings, hotkey, modifiers, openAtPosition, showToolbarByDefault, position, optionLabel, windowType, addToOptionsDict, false) { }

            public FullscreenOption(EditorFullscreenSettings settings, KeyCode hotkey, EventModifiers modifiers, OpenFullscreenAtPosition openAtPosition, bool showToolbarByDefault, Vector2 position, string optionLabel, Type windowType, bool addToOptionsDict, bool isGameView)
            {
                this.hotkey = hotkey;
                this.modifiers = modifiers;
                this.openAtPosition = openAtPosition;
                this.showToolbarByDefault = showToolbarByDefault;
                this.position = position;
                this.optionLabel = optionLabel;
                this.windowTypeAssemblyQualifiedName = windowType == null ? null : windowType.AssemblyQualifiedName;
                this.isGameView = isGameView;
                gameViewOptions = isGameView ? new GameViewOptions() : null;
                AssignOptionID(settings, addToOptionsDict);
            }

            public Type WindowType
            {
                get { return windowTypeAssemblyQualifiedName == null ? null : Type.GetType(windowTypeAssemblyQualifiedName); }
                set { windowTypeAssemblyQualifiedName = value.AssemblyQualifiedName; }
            }
        }

        [Serializable]
        public class GameViewOptions
        {
            public int display;
            public bool lowResolutionAspectRatios;
            public int aspectRatio;
            public float scale;
            public bool stats;
            public bool gizmos;

            public GameViewOptions()
            {
                display = 0;
                lowResolutionAspectRatios = false;
                aspectRatio = 0;
                scale = 1;
            }

            public GameViewOptions(int display, bool lowResolutionAspectRatios, int aspectRatio, float scale)
            {
                this.display = display;
                this.lowResolutionAspectRatios = lowResolutionAspectRatios;
                this.aspectRatio = aspectRatio;
                this.scale = scale;
            }
        }
    }
}