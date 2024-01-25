//AUTO-GENERATED SCRIPT. ANY MODIFICATIONS WILL BE OVERWRITTEN.
//GUID: {b6853174-f1df-4b13-9cd6-7e14c6d64607}
using UnityEngine;
using UnityEditor;
namespace EditorWindowFullscreen {
    public class EditorFullscreenControllerMenuItems {
        [MenuItem(EditorFullscreenSettings.MENU_ITEM_PATH + "Toggle Main Window Fullscreen _F8", false, 21)]
        public static void ToggleMainWindowFullscreen() {EditorFullscreenController.TriggerFullscreenHotkey(EditorFullscreenSettings.settings.mainUnityWindow);}
        [MenuItem(EditorFullscreenSettings.MENU_ITEM_PATH + "Toggle Scene View Fullscreen _F10", false, 22)]
        public static void ToggleSceneViewFullscreen() {EditorFullscreenController.TriggerFullscreenHotkey(EditorFullscreenSettings.settings.sceneWindow);}
        [MenuItem(EditorFullscreenSettings.MENU_ITEM_PATH + "Toggle Game Fullscreen _F11", false, 23)]
        public static void ToggleGameViewFullscreen() {EditorFullscreenController.TriggerFullscreenHotkey(EditorFullscreenSettings.settings.gameWindow);}
        [MenuItem(EditorFullscreenSettings.MENU_ITEM_PATH + "Toggle Focused Window Fullscreen _F9", false, 24)]
        public static void ToggleFocusedWindowFullscreen() {EditorFullscreenController.TriggerFullscreenHotkey(EditorFullscreenSettings.settings.currentlyFocusedWindow);}
        [MenuItem(EditorFullscreenSettings.MENU_ITEM_PATH + "Toggle Mouseover Window Fullscreen _%F9", false, 25)]
        public static void ToggleWindowUnderCursorFullscreen() {EditorFullscreenController.TriggerFullscreenHotkey(EditorFullscreenSettings.settings.windowUnderCursor);}
        [MenuItem(EditorFullscreenSettings.MENU_ITEM_PATH + "Show ∕ Hide Toolbar in Fullscreen _F12", false, 26)]
        public static void ToggleTopToolbar() {EditorFullscreenController.TriggerFullscreenHotkey(EditorFullscreenSettings.settings.toggleTopToolbar);}
        [MenuItem(EditorFullscreenSettings.MENU_ITEM_PATH + "Close all Editor Fullscreen Windows _%F8", false, 27)]
        public static void CloseAllEditorFullscreenWindows() {EditorFullscreenController.TriggerFullscreenHotkey(EditorFullscreenSettings.settings.closeAllFullscreenWindows);}
        [MenuItem(EditorFullscreenSettings.MENU_ITEM_PATH + "Reset to Default Layout _%#&F8", false, 28)]
        public static void ResetToDefaultLayout() {EditorFullscreenController.TriggerFullscreenHotkey(EditorFullscreenSettings.settings.resetToDefaultLayout);}
    }
}
