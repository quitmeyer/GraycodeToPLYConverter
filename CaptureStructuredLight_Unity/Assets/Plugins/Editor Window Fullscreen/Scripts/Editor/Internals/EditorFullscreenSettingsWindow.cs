/* 
 * Author:  Johanan Round
 * Package: Editor Window Fullscreen
 * License: Unity Asset Store EULA (Editor extension asset. Requires 1 license per machine.)
 */

using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;

namespace EditorWindowFullscreen
{
    public class EditorFullscreenSettingsWindow : EditorWindow
    {
        public static EditorFullscreenSettingsWindow window;

#if UNITY_EDITOR_OSX
        [MenuItem(EditorFullscreenSettings.MENU_ITEM_PATH + "Fullscreen Window Settings...\t\t\t\t⇧⌘F8 %#F8", false, 0)]
#else
        [MenuItem(EditorFullscreenSettings.MENU_ITEM_PATH + "Fullscreen Window Settings... %#F8", false, 0)]
#endif
        public static void FullscreenWindowSettings()
        {
            var settingsWin = EditorWindow.GetWindow<EditorFullscreenSettingsWindow>(true, "Editor Window Fullscreen Settings", true);
            try
            {
                //Move the settings window to an offset from the main window
                var mainWinPos = EditorMainWindow.position;
                var newPos = new Rect(settingsWin.position);
                newPos.x = mainWinPos.x + mainWinPos.width / 2 - settingsWin.minSize.x / 2;
                newPos.y = mainWinPos.y + mainWinPos.height / 2 - settingsWin.minSize.y / 2;

                newPos.width = settingsWin.minSize.x;
                newPos.height = settingsWin.minSize.y;

                settingsWin.position = newPos;
            }
            catch
            {
                Debug.Log("Couldn't get the Main Window position");
            }
        }

        private const string GAME_WINDOW_LABEL = "Game Window";
        private const int SETTINGS_CLOSED_AFTER_REV_CHECK_LIMIT = 5;

        private List<bool> foldoutCustomGameWindowSettings = new List<bool>();
        private bool guiVisibilityChange = false;
        Vector2 customGameViewsScrollPos;

        private GUIStyle headerStyle = new GUIStyle();
        private GUIStyle subHeaderStyle = new GUIStyle();
        private GUIStyle smallHeadingStyle = new GUIStyle();
        Color proHeadingColor = new Color(0.75f, 0.75f, 0.75f, 1f);

        public EditorFullscreenSettings _windowSettings = null;
        public EditorFullscreenSettings settings
        {
            get
            {
                if (_windowSettings == null) _windowSettings = EditorFullscreenSettings.LoadSettings();
                return _windowSettings;
            }
        }

        private bool allowApplySettings = false;
        private bool hotkeyWasSet = false;
        private bool hotkeysWereChanged = false;
        private bool customGameViewWasChanged = false;

        private int hotkeyID;
        private string focusedFieldName;
        private List<string> hotkeyInputFields = new List<string>();

        int selectedToolbarItem;

        void ResetSettingsToDefaults()
        {
            EditorFullscreenSettings.ResetToDefaults();
            hotkeysWereChanged = true;
            EditorFullscreenSettings.settings.SaveSettings(hotkeysWereChanged);
            _windowSettings = EditorFullscreenSettings.LoadSettings();
            allowApplySettings = false;
            hotkeysWereChanged = false;
        }

        void OnEnable()
        {
            window = this;
            this.minSize = new Vector2(540, 670);

            headerStyle.fontSize = 18;
            headerStyle.fontStyle = FontStyle.Bold;
            headerStyle.normal.textColor = EditorGUIUtility.isProSkin ? proHeadingColor : new Color(0.25f, 0.25f, 0.25f, 1f);
            headerStyle.margin.top = 10;
            headerStyle.margin.bottom = 5;

            subHeaderStyle.fontSize = 14;
            subHeaderStyle.fontStyle = FontStyle.Bold;
            subHeaderStyle.normal.textColor = EditorGUIUtility.isProSkin ? proHeadingColor : new Color(0.25f, 0.25f, 0.25f, 1f);

            subHeaderStyle.margin.top = 10;

            smallHeadingStyle.fontStyle = FontStyle.Bold;
            smallHeadingStyle.margin.top = 5;
            smallHeadingStyle.margin.left = 6;
            if (EditorGUIUtility.isProSkin)
                smallHeadingStyle.normal.textColor = proHeadingColor;

            _windowSettings = EditorFullscreenSettings.LoadSettings();
            EditorFullscreenState.TriggerFullscreenEvent(this, this.GetType(), Vector2.zero, false); //Notify everyone that the settings window was opened.
        }

        void OnDisable()
        {
            if (settings.clickedLeaveRevCount > 0 && settings.settingsClosedAfterRevCount < SETTINGS_CLOSED_AFTER_REV_CHECK_LIMIT)
            {
                EditorFullscreenSettings.settings.settingsClosedAfterRevCount++;
                EditorFullscreenSettings.settings.SaveSettings(false);
            }
        }
        void OnGUI()
        {
            hotkeyWasSet = false;
            hotkeyInputFields.Clear();
            hotkeyID = 0;
            var style = new GUIStyle();
            var buttonStyle = new GUIStyle();
            var smallIndent = new GUIStyle();
            style.margin.left = 20;
            style.margin.right = 20;
            smallIndent.margin.left = 3;

            //Toolbar
            EditorGUILayout.BeginVertical(style);
            var headerToolbarStyle = new GUIStyle(GUI.skin.button);
            headerToolbarStyle.fontSize = 15;
            headerToolbarStyle.fontStyle = FontStyle.Bold;
            headerToolbarStyle.normal.textColor = EditorGUIUtility.isProSkin ? proHeadingColor : new Color(0.25f, 0.25f, 0.25f, 1f);
            headerToolbarStyle.margin.top = 10;
            headerToolbarStyle.margin.bottom = 5;
            string[] toolbarItems = { "Fullscreen Options", "Game Window Options" };

            EditorGUI.BeginChangeCheck();
            selectedToolbarItem = GUILayout.Toolbar(selectedToolbarItem, toolbarItems, headerToolbarStyle, GUILayout.Height(32));
            guiVisibilityChange = EditorGUI.EndChangeCheck() || guiVisibilityChange;

            EditorGUILayout.EndVertical();

            //Page header
            EditorGUILayout.BeginVertical(style);
            EditorGUILayout.BeginHorizontal();
            EditorGUIUtility.labelWidth = 245f;
            GUILayout.Label(toolbarItems[selectedToolbarItem], headerStyle, new GUILayoutOption[0]);

            //Reset to defaults button
            buttonStyle = new GUIStyle(GUI.skin.button);
            buttonStyle.fontSize = 10;
            buttonStyle.fontStyle = FontStyle.Bold;
            buttonStyle.padding = new RectOffset(0, 0, 5, 5);
            buttonStyle.margin.top = 10;
            buttonStyle.margin.right = 24;
            if (GUILayout.Button(new GUIContent("Reset to Defaults", "Reset all settings to their default values."), buttonStyle, GUILayout.MaxWidth(200)))
            {
                ResetSettingsToDefaults();
                Repaint();
                return;
            }
            EditorGUILayout.EndHorizontal();

            //Indent
            EditorGUILayout.BeginVertical(style);

            //Debugging checkbox
            EditorGUILayout.BeginHorizontal();
            var s = new GUIStyle(GUI.skin.toggle);
            s.alignment = TextAnchor.LowerRight;
            s.normal.textColor = EditorStyles.label.normal.textColor;
            s.padding.top = -5;
            s.margin.top = -5;
            EditorGUIUtility.labelWidth = 0.1f;

            GUILayout.FlexibleSpace();
            var dbStyle = new GUIStyle();
            dbStyle.fontSize = 9;
            dbStyle.normal.textColor = EditorStyles.label.normal.textColor;
            dbStyle.alignment = TextAnchor.LowerRight;
            dbStyle.fixedHeight = 15f;
            GUILayout.Label("Debugging", dbStyle, new GUILayoutOption[0]);
            settings.debugModeEnabled = EditorGUILayout.Toggle(new GUIContent(" ", "Enables debug mode"), settings.debugModeEnabled, s);

            EditorGUILayout.EndHorizontal();
            GUILayout.Space(-10);

            if (selectedToolbarItem == 0)
            {
                //Fullscreen Window Hotkeys
                EditorGUIUtility.labelWidth = 175f;
                GUILayout.Label("Fullscreen Window Hotkeys", subHeaderStyle, new GUILayoutOption[0]);
                EditorGUILayout.BeginVertical(smallIndent);
#if UNITY_EDITOR_OSX
                bool showToolbarOptionForMainUnityWindow = false; //Toolbar option isn't necessary for main window on Mac (toolbar can be toggled by moving the mouse to the top of the screen).
#else
                bool showToolbarOptionForMainUnityWindow = true;
#endif
                AddFullscreenOption(ref settings.mainUnityWindow, "Main Unity Window", true, showToolbarOptionForMainUnityWindow);
                AddFullscreenOption(ref settings.sceneWindow, "Scene Window", true, true);
                AddFullscreenOption(ref settings.gameWindow, GAME_WINDOW_LABEL, true, true, true);
                AddFullscreenOption(ref settings.currentlyFocusedWindow, "Currently Focused Window", true, true);
                AddFullscreenOption(ref settings.windowUnderCursor, "Window Under Cursor", true, true);
                EditorGUILayout.EndVertical();

                //Other Hotkeys
                GUILayout.Label("Other Options", subHeaderStyle, new GUILayoutOption[0]);
                EditorGUILayout.BeginVertical(smallIndent);
                AddFullscreenOption(ref settings.toggleTopToolbar, "Show/Hide Top Toolbar", false, false);
                EditorGUILayout.EndVertical();
                style = new GUIStyle(smallIndent);
                style.margin.top = 8;
                EditorGUILayout.BeginVertical(style);
                AddFullscreenOption(ref settings.closeAllFullscreenWindows, "Close All Fullscreen Windows", false, false);
                EditorGUILayout.EndVertical();

                EditorGUILayout.BeginVertical(style);
                AddFullscreenOption(ref settings.resetToDefaultLayout, "Reset to Default Layout", false, false);
                EditorGUILayout.EndVertical();

                EditorGUILayout.BeginVertical(style);
                EditorGUILayout.BeginHorizontal();

                EditorGUIUtility.labelWidth = 255f;
                var label = new GUIContent("Show a Notification on Fullscreen Entry", "Show a notification when entering fullscreen.");
                style = new GUIStyle(smallHeadingStyle);

                GUILayout.Label(label, style, new GUILayoutOption[0]);

                style = new GUIStyle();
                style.alignment = TextAnchor.LowerRight;
                EditorGUILayout.BeginHorizontal(new[] { GUILayout.MaxWidth(160f) });
                settings.fullscreenNotification = EditorGUILayout.Toggle(settings.fullscreenNotification, new GUILayoutOption[0]);
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.EndHorizontal();
                EditorGUILayout.EndVertical();
            }
            else if (selectedToolbarItem == 1)
            {
                //Game Window Fullscreen Options
                EditorGUIUtility.labelWidth = 255f;
                GUILayout.Label("Game Start/Stop", subHeaderStyle, new GUILayoutOption[0]);
                EditorGUILayout.BeginVertical(smallIndent);

                var label = new GUIContent("Start the Game When Entering Fullscreen", "Start the game when entering a fullscreen game window.");
                settings.startGameWhenEnteringFullscreen = EditorGUILayout.Toggle(label, settings.startGameWhenEnteringFullscreen, new GUILayoutOption[0]);

                label = new GUIContent("Stop the Game When Exiting Fullscreen", "Stop the game when exiting fullscreen game window.");
                settings.stopGameWhenExitingFullscreen = (EditorFullscreenSettings.StopGameWhenExitingFullscreen)EditorGUILayout.EnumPopup(label, settings.stopGameWhenExitingFullscreen, new GUILayoutOption[0]);

                label = new GUIContent("Close Fullscreen On Game Stop", "Close fullscreen game window/s when the game stops.");
                settings.closeFullscreenOnGameStop = (EditorFullscreenSettings.CloseFullscreenOnGameStop)EditorGUILayout.EnumPopup(label, settings.closeFullscreenOnGameStop, new GUILayoutOption[0]);

                GUILayout.Label("Framerate", subHeaderStyle, new GUILayoutOption[0]);
                label = new GUIContent("To Improve FPS When Entering Fullscreen", "Multiple visible game views can significantly affect the performance/FPS of the game. Choose an option to prevent this from happening.");
                settings.improveFpsOptions = (EditorFullscreenSettings.ImproveFPSOptions)EditorGUILayout.EnumPopup(label, settings.improveFpsOptions, new GUILayoutOption[0]);

                //Custom Fullscreen Game Views
                GUILayout.Label("Custom Fullscreen Game Views", subHeaderStyle, new GUILayoutOption[0]);

                EditorGUIUtility.labelWidth = 184f;
                customGameViewsScrollPos = EditorGUILayout.BeginScrollView(customGameViewsScrollPos, false, true, GUIStyle.none, GUI.skin.verticalScrollbar, GUIStyle.none, new GUILayoutOption[0]);
                var groupVertStyle = new GUIStyle();
                groupVertStyle.padding.right = 20;
                groupVertStyle.margin.right = 5;
                EditorGUILayout.BeginVertical(groupVertStyle, GUILayout.MinHeight(360));

#if UNITY_2019_1_OR_NEWER
                var foldoutStyle = new GUIStyle(EditorStyles.foldoutHeader);
                foldoutStyle.margin.bottom = 5;
                foldoutStyle.margin.top = 15;
#else
                var foldoutStyle = new GUIStyle(EditorStyles.foldout);
                foldoutStyle.fontStyle = FontStyle.Bold;
                var dropLineStyle = new GUIStyle();
                dropLineStyle.margin.top = 10;
                dropLineStyle.margin.bottom = 0;
                var indentStyle = new GUIStyle();
                indentStyle.margin.left = 12;
                var popupStyle = GUI.skin.FindStyle("IconButton");
                var popupIcon = EditorGUIUtility.IconContent("_Popup");
#endif

                var allGameWins = settings.AllGameWindows;
                for (int i = 0; i < allGameWins.Count; i++)
                {
                    if (foldoutCustomGameWindowSettings.Count < (i + 1))
                    {
                        foldoutCustomGameWindowSettings.Add(i == 0 ? true : false);
                    }
                    var fullscreenOption = allGameWins[i];

                    EditorGUI.BeginChangeCheck();
#if UNITY_2019_1_OR_NEWER
                    foldoutCustomGameWindowSettings[i] = EditorGUILayout.BeginFoldoutHeaderGroup(foldoutCustomGameWindowSettings[i], fullscreenOption.optionLabel + (i == 0 || fullscreenOption.optionLabel != "Game Window" ? "" : " " + (i + 1)), foldoutStyle, menuPos => ShowGameWindowFoldoutMenu(menuPos, i));
                    guiVisibilityChange = EditorGUI.EndChangeCheck() || guiVisibilityChange;
                    if (foldoutCustomGameWindowSettings[i]) AddFullscreenOption(ref fullscreenOption, null, true, true);
                    EditorGUILayout.EndFoldoutHeaderGroup();
#else
                    EditorGUILayout.BeginHorizontal(dropLineStyle);
                    foldoutCustomGameWindowSettings[i] = EditorGUILayout.Foldout(foldoutCustomGameWindowSettings[i], fullscreenOption.optionLabel + (i == 0 || fullscreenOption.optionLabel != "Game Window" ? "" : " " + (i + 1)), true, foldoutStyle);
                    var foldoutRect = EditorGUILayout.GetControlRect(false, 20f, GUILayout.MaxWidth(20f));
                    EditorGUILayout.EndHorizontal();
                    guiVisibilityChange = EditorGUI.EndChangeCheck() || guiVisibilityChange;
                    if (popupStyle == null) popupStyle = EditorStyles.popup;
                    
                    if (GUI.Button(foldoutRect, popupIcon, popupStyle))
                    {
                        ShowGameWindowFoldoutMenu(foldoutRect, i);
                    }
                    EditorGUILayout.BeginVertical(indentStyle);
                    if (foldoutCustomGameWindowSettings[i]) AddFullscreenOption(ref fullscreenOption, null, true, true);
                    EditorGUILayout.EndVertical();
#endif
                }

                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                buttonStyle = new GUIStyle(GUI.skin.button);
                buttonStyle.fontSize = 12;
                buttonStyle.padding = new RectOffset(10, 10, 4, 4);
                buttonStyle.margin.top = 12;
                if (GUILayout.Button(new GUIContent("Add Game View", "Add a custom game view with its own settings and hotkey."), buttonStyle, GUILayout.MaxWidth(250)))
                {
                    //Add a Game View
                    if (settings.customWindows == null) settings.customWindows = new List<EditorFullscreenSettings.FullscreenOption>();
                    var fullscreenOption = new EditorFullscreenSettings.FullscreenOption(settings, true);
                    fullscreenOption.windowTypeAssemblyQualifiedName = EditorFullscreenState.GameViewType.AssemblyQualifiedName;
                    fullscreenOption.optionLabel = "Game Window";
                    fullscreenOption.openAtPosition = EditorFullscreenSettings.OpenFullscreenAtPosition.AtSpecifiedPosition;
                    fullscreenOption.gameViewOptions = new EditorFullscreenSettings.GameViewOptions();
                    fullscreenOption.isGameView = true;
                    settings.customWindows.Add(fullscreenOption);
                    foldoutCustomGameWindowSettings.Add(true);
                }
                buttonStyle.alignment = TextAnchor.LowerCenter;
                GUILayout.FlexibleSpace();
                EditorGUILayout.EndHorizontal(); //End Add Game View button horizontal
                EditorGUILayout.EndVertical(); //End Add Game View vertical scroll area
                EditorGUILayout.EndScrollView();

                EditorGUILayout.EndVertical();
            }

            if ((GUI.changed && !guiVisibilityChange) || hotkeyWasSet || customGameViewWasChanged)
            {
                if (hotkeyWasSet) hotkeysWereChanged = true;
                allowApplySettings = true;
                if (customGameViewWasChanged) customGameViewWasChanged = false;
            }
            if (guiVisibilityChange) guiVisibilityChange = false;

            GUILayout.FlexibleSpace();
            EditorGUILayout.BeginVertical();

            var lineStyle = new GUIStyle();
            lineStyle.margin.top = 10;
            EditorGUILayout.BeginHorizontal(lineStyle);
            try
            {
                var linkLabel = typeof(EditorGUILayout).GetMethod("LinkLabel", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static, null, new System.Type[] { typeof(GUIContent), typeof(GUILayoutOption[]) }, null);
                if (settings.fullscreenedCount > 19 && settings.clickedLeaveRevCount < 3 && (settings.clickedLeaveRevCount < 1 || settings.settingsClosedAfterRevCount < SETTINGS_CLOSED_AFTER_REV_CHECK_LIMIT))
                {
                    var msgStyle = new GUIStyle();
                    msgStyle.fontStyle = FontStyle.Normal;
                    msgStyle.margin.top = 2;
                    msgStyle.margin.right = 0;
                    if (EditorGUIUtility.isProSkin) msgStyle.normal.textColor = new Color(0.75f, 0.75f, 0.75f, 1f);
                    lineStyle = new GUIStyle();
                    lineStyle.margin.right = 6;
                    EditorGUILayout.EndHorizontal();
                    EditorGUILayout.BeginHorizontal(lineStyle);
                    GUILayout.FlexibleSpace();
                    GUILayout.Label("Thanks for using this extension!", msgStyle, new GUILayoutOption[0]);
                    lineStyle = new GUIStyle();
                    lineStyle.margin.bottom = 10;
                    EditorGUILayout.EndHorizontal();
                    EditorGUILayout.BeginHorizontal(lineStyle);
                    GUILayout.FlexibleSpace();
                    var msgStyle2 = new GUIStyle(msgStyle);
                    #if UNITY_2019_3_OR_NEWER
                    msgStyle2.margin.top = 5;
                    #endif
                    GUILayout.Label("If you find it useful, please consider", msgStyle2, new GUILayoutOption[] {
                    #if UNITY_EDITOR_OSX
                    #if UNITY_2019_3_OR_NEWER
                    GUILayout.MaxWidth(194f)
                    #else
                    GUILayout.MaxWidth(188f)
                    #endif
                    #else
                    #if UNITY_2019_3_OR_NEWER
                    GUILayout.MaxWidth(203f)
                    #else
                    GUILayout.MaxWidth(200f)
                    #endif
                    #endif
                    });
                    var leaveRevClicked = (bool)linkLabel.Invoke(null, new object[] { new GUIContent("giving us a review."), new GUILayoutOption[] { } });
                    if (leaveRevClicked)
                    {
                        Application.OpenURL("https://assetstore.unity.com/packages/tools/utilities/editor-window-fullscreen-85477");
                        settings.clickedLeaveRevCount++;
                        EditorFullscreenSettings.settings.clickedLeaveRevCount++;
                        EditorFullscreenSettings.settings.SaveSettings(false);
                    }
                    EditorGUILayout.EndHorizontal();
                    EditorGUILayout.BeginHorizontal();
                }
                var website = new GUIContent("Official Website");
                var supportThread = new GUIContent("Forum Support & Info Thread");
                var supportEmail = new GUIContent("support@crystalconflux.com");
                var websiteClicked = (bool)linkLabel.Invoke(null, new object[] { website, new GUILayoutOption[0] });
                if (websiteClicked)
                {
                    Application.OpenURL("https://crystalconflux.com/unity-editor-extensions/editor-window-fullscreen/");
                }
                GUILayout.FlexibleSpace();
                var supportThreadClicked = (bool)linkLabel.Invoke(null, new object[] { supportThread, new GUILayoutOption[0] });
                if (supportThreadClicked)
                {
                    Application.OpenURL("https://forum.unity3d.com/threads/473619/");
                }
                GUILayout.FlexibleSpace();
                var supportEmailClicked = (bool)linkLabel.Invoke(null, new object[] { supportEmail, new GUILayoutOption[0] });
                if (supportEmailClicked)
                {
                    var subj = "Editor Window Fullscreen - Support Request - (My OS: " + EWFDebugging.OSVersion + ", Unity: " + Application.unityVersion + ", EWF: " + EditorFullscreenSettings.Version + ")";
                    subj = subj.Replace(" ", "%20").Replace(":", "%3A").Replace(",", "%2C");
                    Application.OpenURL("mailto:support@crystalconflux.com?subject=" + subj);
                }
            }
            catch (System.Exception e) { if (EWFDebugging.Enabled) Debug.LogWarning("Couldn't display EWF links.\n" + e); }
            EditorGUILayout.EndHorizontal();

            EditorGUI.BeginDisabledGroup(!allowApplySettings);
            buttonStyle = new GUIStyle(GUI.skin.button);
            buttonStyle.fontSize = 15;
            buttonStyle.fontStyle = FontStyle.Bold;
            buttonStyle.padding = new RectOffset(0, 0, 10, 10);
            buttonStyle.margin.top = 7;
            buttonStyle.alignment = TextAnchor.LowerCenter;
            if (GUILayout.Button(new GUIContent("Apply Settings", "Apply all changes."), buttonStyle, new GUILayoutOption[0]))
            {
                settings.SaveSettings(hotkeysWereChanged);
                EditorFullscreenSettings.ReloadSettings();
                allowApplySettings = false;
                hotkeysWereChanged = false;
            }
            EditorGUI.EndDisabledGroup();

            EditorGUI.BeginDisabledGroup(true);
            style = new GUIStyle();
            style.fontStyle = FontStyle.Normal;
            style.fontSize = 9;
            style.margin.left = 9;
            style.margin.top = 6;
            style.margin.bottom = 6;
            if (EditorGUIUtility.isProSkin) style.normal.textColor = new Color(0.75f, 0.75f, 0.75f, 1f);

            GUILayout.Label("Current Mouse Position: " + EditorInput.MousePosition, style, new GUILayoutOption[0]);
            EditorGUI.EndDisabledGroup();

            //Set the focused control
            if (Event.current.type == EventType.Repaint)
            {
                focusedFieldName = GUI.GetNameOfFocusedControl();
            }
            if (hotkeyWasSet && Event.current.type == EventType.KeyDown) Event.current.Use();

            EditorGUILayout.EndVertical();
            EditorGUILayout.EndVertical();
            EditorGUILayout.EndVertical();
        }

        void ShowGameWindowFoldoutMenu(Rect menuPosition, int gameWinIndex)
        {
            var popupMenu = new GenericMenu();
            if (gameWinIndex == 0)
            {
                popupMenu.AddItem(new GUIContent("Primary (Unremovable)"), true, () => RemoveCustomGameViewItem(gameWinIndex));
                popupMenu.DropDown(menuPosition);
            }
            else
            {
                popupMenu.AddItem(new GUIContent("Rename"), false, () => RenameCustomGameViewItem(menuPosition, gameWinIndex));
                popupMenu.AddItem(new GUIContent("Remove"), false, () => RemoveCustomGameViewItem(gameWinIndex));
                popupMenu.DropDown(menuPosition);
            }
        }

        void RenameCustomGameViewItem(Rect menuPosition, int gameWinIndex)
        {
            var fullscreenOps = settings.AllGameWindows[gameWinIndex];
            if (fullscreenOps == null) return;
            var renameWin = EditorTextEntryWindow.Create("Choose a Name", fullscreenOps.optionLabel, entryText => SaveNewNameGameViewItemName(fullscreenOps.OptionID, entryText));
            renameWin.ShowAt(position.center, true);
        }

        void SaveNewNameGameViewItemName(int optionID, string customGameViewName)
        {
            var fullscreenOps = settings.GetFullscreenOption(optionID);
            if (fullscreenOps != null)
                fullscreenOps.optionLabel = customGameViewName == null ? "Game Window" : customGameViewName;
            customGameViewWasChanged = true;
            hotkeysWereChanged = true;
        }

        void RemoveCustomGameViewItem(int gameWinIndex)
        {
            if (gameWinIndex == 0 || settings.customWindows == null) return;
            int optionID = settings.AllGameWindows[gameWinIndex].OptionID;
            settings.customWindows.RemoveAt(gameWinIndex - 1);
            foldoutCustomGameWindowSettings.RemoveAt(gameWinIndex);
            customGameViewWasChanged = true;
            hotkeysWereChanged = true;
        }

        private void SetAspectRatio(int optionID, int aspectRatioIndex)
        {
            settings.GetFullscreenOption(optionID).gameViewOptions.aspectRatio = aspectRatioIndex;
            customGameViewWasChanged = true;
        }

        void AddFullscreenOption(ref EditorFullscreenSettings.FullscreenOption fullscreenOption, string label, bool showOpenAtOption, bool showToolbarOption)
        {
            AddFullscreenOption(ref fullscreenOption, label, showOpenAtOption, showToolbarOption, false);
        }

        void AddFullscreenOption(ref EditorFullscreenSettings.FullscreenOption fullscreenOption, string label, bool showOpenAtOption, bool showToolbarOption, bool hideGameViewExtraOptions)
        {
            AddFullscreenOption(ref fullscreenOption, (label == null ? null : new GUIContent(label)), showOpenAtOption, showToolbarOption, true, new[] { 1, 2, 3 }, hideGameViewExtraOptions);
        }

        void AddFullscreenOption(ref EditorFullscreenSettings.FullscreenOption fullscreenOption, string label, bool showOpenAtOption, bool showToolbarOption, int[] displayedOpenAtOptions)
        {
            AddFullscreenOption(ref fullscreenOption, (label == null ? null : new GUIContent(label)), showOpenAtOption, showToolbarOption, true, displayedOpenAtOptions, false);
        }

        void AddFullscreenOption(ref EditorFullscreenSettings.FullscreenOption fullscreenOption, GUIContent label, bool showOpenAtOption, bool showToolbarOption, bool showHotkey, int[] displayedOpenAtOptions, bool hideGameViewExtraOptions)
        {
            var indent = new GUIStyle();
            if (!showHotkey) indent.margin.left = 10;
            var initLabelWidth = EditorGUIUtility.labelWidth;

            if (!(showOpenAtOption || showToolbarOption))
                EditorGUILayout.BeginHorizontal();

            if (label != null) GUILayout.Label(label, smallHeadingStyle, new GUILayoutOption[0]);

            if (showOpenAtOption || showToolbarOption)
            {
                EditorGUIUtility.labelWidth = initLabelWidth - indent.margin.left;
                EditorGUILayout.BeginVertical(indent);
            }

            if (showOpenAtOption)
            {
                EditorGUILayout.BeginHorizontal();
                fullscreenOption.openAtPosition = (EditorFullscreenSettings.OpenFullscreenAtPosition)AddFilteredEnumPopup(new GUIContent("Enter Fullscreen"), fullscreenOption.openAtPosition, displayedOpenAtOptions);

                if (fullscreenOption.openAtPosition == EditorFullscreenSettings.OpenFullscreenAtPosition.AtSpecifiedPosition)
                {
                    var prevLabelWidth = EditorGUIUtility.labelWidth;
                    EditorGUIUtility.labelWidth = 15;
                    EditorGUILayout.BeginHorizontal(new[] { GUILayout.MaxWidth(100f) });
                    fullscreenOption.position.x = EditorGUILayout.IntField(new GUIContent("x", "x position"), (int)fullscreenOption.position.x, new GUILayoutOption[0]);
                    fullscreenOption.position.y = EditorGUILayout.IntField(new GUIContent("y", "y position"), (int)fullscreenOption.position.y, new GUILayoutOption[0]);
                    EditorGUILayout.EndHorizontal();
                    EditorGUIUtility.labelWidth = prevLabelWidth;
                }
                EditorGUILayout.EndHorizontal();
            }
            if (showOpenAtOption || showToolbarOption)
                EditorGUILayout.BeginHorizontal();
            if (showToolbarOption)
                fullscreenOption.showToolbarByDefault = EditorGUILayout.Toggle("Show Toolbar By Default", fullscreenOption.showToolbarByDefault, new GUILayoutOption[0]);

            //Hotkey
            if (showHotkey)
            {
                if (!showToolbarOption) GUILayout.FlexibleSpace();
                var hotkeySet = AddHotkeyField("Hotkey", ref fullscreenOption.hotkey, ref fullscreenOption.modifiers, fullscreenOption.OptionID);
                hotkeyWasSet = hotkeyWasSet || hotkeySet;
            }
            EditorGUILayout.EndHorizontal();

            if (fullscreenOption.isGameView && !hideGameViewExtraOptions)
            {
                //Game View specific settings
                if (fullscreenOption.gameViewOptions == null) fullscreenOption.gameViewOptions = new EditorFullscreenSettings.GameViewOptions();
                var lineStyle = new GUIStyle();
                lineStyle.margin.top = 5;
                lineStyle.margin.right = 25;

                //Open Fullscreen On Game Start
                EditorGUILayout.BeginHorizontal(lineStyle);
                fullscreenOption.openOnGameStart = EditorGUILayout.Toggle("Open Fullscreen On Game Start", fullscreenOption.openOnGameStart, new GUILayoutOption[0]);
                EditorGUILayout.EndHorizontal();

                //Game Display
                EditorGUILayout.BeginHorizontal(lineStyle);
                EditorGUILayout.LabelField("Game Display", GUILayout.MaxWidth(EditorGUIUtility.labelWidth - 4));
                GUIContent[] displayNames = null;
                if (EditorFullscreenState.DisplayUtilityType != null)
                {
                    var getDisplayNames = EditorFullscreenState.DisplayUtilityType.GetMethod("GetDisplayNames", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
                    if (getDisplayNames != null) displayNames = (GUIContent[])getDisplayNames.Invoke(null, null);
                }
                if (displayNames == null) displayNames = new GUIContent[] { new GUIContent("Display 1"), new GUIContent("Display 2"), new GUIContent("Display 3"), new GUIContent("Display 4"), new GUIContent("Display 5"), new GUIContent("Display 6"), new GUIContent("Display 7"), new GUIContent("Display 8") };
                fullscreenOption.gameViewOptions.display = EditorGUILayout.Popup(fullscreenOption.gameViewOptions.display, displayNames);
                EditorGUILayout.EndHorizontal();

                //Low Resolution Aspect Ratios
                EditorGUILayout.BeginHorizontal(lineStyle);
                fullscreenOption.gameViewOptions.lowResolutionAspectRatios = EditorGUILayout.Toggle("Low Resolution Aspect Ratios", fullscreenOption.gameViewOptions.lowResolutionAspectRatios, new GUILayoutOption[0]);
                EditorGUILayout.EndHorizontal();

                //Aspect Ratio
                lineStyle.margin.top = 5;
                lineStyle.margin.right += 5;
                EditorGUILayout.BeginHorizontal(lineStyle);
                EditorGUILayout.LabelField("Aspect Ratio", GUILayout.MaxWidth(EditorGUIUtility.labelWidth - 4));
                if (EditorFullscreenState.GameViewType != null)
                {
                    AspectRatioPopup(fullscreenOption);
                }
                EditorGUILayout.EndHorizontal();
                lineStyle.margin.right -= 5;

                //Scale
                EditorGUILayout.BeginHorizontal(lineStyle);
                EditorGUILayout.LabelField("Scale", GUILayout.MaxWidth(EditorGUIUtility.labelWidth - 4));
                fullscreenOption.gameViewOptions.scale = EditorGUILayout.Slider(fullscreenOption.gameViewOptions.scale, 1, 12f);
                EditorGUILayout.EndHorizontal();

                //Show Stats
                lineStyle.margin.top = 0;
                EditorGUILayout.BeginHorizontal(lineStyle);
                fullscreenOption.gameViewOptions.stats = EditorGUILayout.Toggle("Show Stats", fullscreenOption.gameViewOptions.stats, new GUILayoutOption[0]);
                EditorGUILayout.EndHorizontal();

                //Show Gizmos
                EditorGUILayout.BeginHorizontal(lineStyle);
                fullscreenOption.gameViewOptions.gizmos = EditorGUILayout.Toggle("Show Gizmos", fullscreenOption.gameViewOptions.gizmos, new GUILayoutOption[0]);
                EditorGUILayout.EndHorizontal();
            }

            if (showOpenAtOption || showToolbarOption)
            {
                EditorGUILayout.EndVertical();
                EditorGUIUtility.labelWidth = initLabelWidth;
            }
        }

        private void AspectRatioPopup(EditorFullscreenSettings.FullscreenOption fullscreenOption)
        {
            try
            {
                var selectedAspectRatioIndex = fullscreenOption.gameViewOptions.aspectRatio;
                var buttonPos = EditorGUILayout.GetControlRect(false, 16f, EditorStyles.toolbarPopup);

                var gameViewSizesInstance = EditorFullscreenState.GameViewSizesType.BaseType.GetProperty("instance");
                var getGroup = EditorFullscreenState.GameViewSizesType.GetMethod("GetGroup", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                var currentSizeGroupType = EditorFullscreenState.GameViewType.GetProperty("currentSizeGroupType", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);

                var gameViewSizes = gameViewSizesInstance.GetValue(null, null);
                var sizeGroupType = currentSizeGroupType.GetValue(null, null);

                var gameViewSizeGroup = getGroup.Invoke(gameViewSizes, new object[] { sizeGroupType });
                var getGroupTotalCount = gameViewSizeGroup.GetType().GetMethod("GetTotalCount");
                int groupTotalCount = (int)getGroupTotalCount.Invoke(gameViewSizeGroup, null);
                var getGameViewSize = gameViewSizeGroup.GetType().GetMethod("GetGameViewSize");
                var currentAspectRatioName = "";
                if (selectedAspectRatioIndex >= 0 && selectedAspectRatioIndex < groupTotalCount)
                {
                    var gameViewSize = getGameViewSize.Invoke(gameViewSizeGroup, new object[] { selectedAspectRatioIndex });
                    var gameViewSizeDisplayText = gameViewSize.GetType().GetProperty("displayText");
                    currentAspectRatioName = (string)gameViewSizeDisplayText.GetValue(gameViewSize, null);
                }

                if (GUI.Button(buttonPos, new GUIContent(currentAspectRatioName), EditorStyles.toolbarPopup))
                {
                    var menuItemProvider = EditorFullscreenState.GameViewSizesMenuItemProvider.GetConstructor(new System.Type[] { sizeGroupType.GetType() }).Invoke(new object[] { sizeGroupType });
                    var menuModifyItemUI = EditorFullscreenState.GameViewSizesMenuModifyItemUI.GetConstructor(new System.Type[] { }).Invoke(new object[] { });
                    var flexibleMenuConstructor = EditorFullscreenState.FlexibleMenuType.GetConstructor(new System.Type[] { menuItemProvider.GetType(), typeof(int), menuModifyItemUI.GetType(), typeof(System.Action<int, object>) });

                    var optionID = fullscreenOption.OptionID;
                    System.Action<int, object> menuItemSelectionCallback = (i, o) => SetAspectRatio(optionID, i);
                    var gameViewAspectRatiosMenu = flexibleMenuConstructor.Invoke(new object[] { menuItemProvider, selectedAspectRatioIndex, menuModifyItemUI, menuItemSelectionCallback });
                    PopupWindow.Show(buttonPos, (PopupWindowContent)gameViewAspectRatiosMenu);
                }
            }
            catch (ExitGUIException e)
            {
                throw e;
            }
            catch (System.Exception e)
            {
                if (EWFDebugging.Enabled) Debug.LogError("Problem while displaying the aspect ratio popup menu: " + e.ToString() + (e.InnerException == null ? "" : "\r\n\r\nInnerException:\r\n" + e.InnerException + "\r\n\r\nOuter Callstack: "));
            }
        }

        private int AddFilteredEnumPopup(GUIContent label, System.Enum selectedValue, int[] displayedOptions)
        {
            var selectedVal = System.Convert.ToInt32(selectedValue);
            var visibleOptions = new GUIContent[displayedOptions.Length];
            var optionValues = new int[displayedOptions.Length];

            for (int i = 0; i < displayedOptions.Length; i++)
            {
                optionValues[i] = displayedOptions[i];
                visibleOptions[i] = new GUIContent();
                visibleOptions[i].text = System.Enum.GetName(selectedValue.GetType(), displayedOptions[i]);
                if (visibleOptions[i].text == null) visibleOptions[i].text = "Undefined";
                else
                {
                    visibleOptions[i].text = EditorFullscreenSettings.FormatCamelCaseName(visibleOptions[i].text);
                }
            }

            return EditorGUILayout.IntPopup(label, selectedVal, visibleOptions, optionValues, new GUILayoutOption[0]);
        }

        public bool IsFocusedOnHotkeyField()
        {
            return hotkeyInputFields.Contains(focusedFieldName);
        }

        public void HotkeyConflictResendEvent(KeyCode keyCode, EventModifiers modifiers)
        {
            var e = new Event();
            e.type = EventType.KeyDown;
            e.keyCode = keyCode;
            e.modifiers = modifiers;
            e.commandName = "HotkeyConflictResendEvent";
            this.SendEvent(e);
        }

        private bool AddHotkeyField(string label, ref KeyCode hotkey, ref EventModifiers modifiers, int optionID)
        {
            bool hotkeyWasSet = false;
            var guiLabel = new GUIContent(label);
            var s = new GUIStyle(GUIStyle.none);
            s.alignment = TextAnchor.LowerRight;
            s.fixedHeight = 15f;
            s.margin.left = smallHeadingStyle.margin.left;
            s.normal.textColor = EditorStyles.label.normal.textColor;

            EditorGUILayout.BeginHorizontal(new[] { GUILayout.MaxWidth(230f) });
            GUILayout.Label(guiLabel, s, new GUILayoutOption[0]);
            Rect textFieldRect = GUILayoutUtility.GetRect(guiLabel, GUI.skin.textField, new GUILayoutOption[0]);

            //Give the control a unique name using its label and a hotkey ID
            hotkeyID++;
            string controlName = label + hotkeyID;
            hotkeyInputFields.Add(controlName);

            if (Event.current.type != EventType.Repaint)
            {
                //Check for Key Press (Must be done before the TextField is set, because it uses the event)
                if (focusedFieldName == controlName && Event.current.type == EventType.KeyDown)
                {
                    if (Event.current.keyCode != KeyCode.None && Event.current.keyCode != KeyCode.LeftControl)
                    {
                        hotkey = Event.current.keyCode;
                        modifiers = Event.current.modifiers;

                        //Clear the hotkey if push Escape, Backspace, or Delete.
                        if ((modifiers == EventModifiers.None || modifiers == EventModifiers.FunctionKey) &&
                            (hotkey == KeyCode.Escape || hotkey == KeyCode.Backspace || hotkey == KeyCode.Delete))
                        {
                            hotkey = KeyCode.None;
                        }

                        hotkeyWasSet = true;

                        settings.ClearHotkeyConflicts(optionID, hotkey, modifiers);
                    }
                }
            }

            //Create the GUI Hotkey Field
            string keysDownString = EditorInput.GetKeysDownString(hotkey, modifiers);
            GUI.SetNextControlName(controlName); //Assign the control its name.
            EditorGUI.SelectableLabel(textFieldRect, "", EditorStyles.textField);
            EditorGUI.LabelField(textFieldRect, keysDownString, EditorStyles.label);
            EditorGUILayout.EndHorizontal();

            return hotkeyWasSet;
        }


        void OnInspectorUpdate()
        {
            Repaint();
        }

    }
}
