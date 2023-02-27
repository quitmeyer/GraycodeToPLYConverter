/* 
 * Author:  Johanan Round
 * Package: Editor Window Fullscreen
 * License: Unity Asset Store EULA (Editor extension asset. Requires 1 license per machine.)
 */

using UnityEngine;
using UnityEditor;

namespace EditorWindowFullscreen
{
    public class EditorTextEntryWindow : EditorWindow
    {
        public string enteredText = "";
        private Vector2 winSize = new Vector2(240, 65);

        public delegate void SaveFunction(string saveText);
        private SaveFunction saveFunction = null;

        private bool initialFocusWasSet = false;

        public static EditorTextEntryWindow Create(string windowTitle, string existingText, SaveFunction saveFunction)
        {
            var createdWin = CreateInstance<EditorTextEntryWindow>();

            if (existingText != null)
                createdWin.enteredText = existingText;

            createdWin.saveFunction = saveFunction;

            createdWin.minSize = createdWin.winSize;
            createdWin.maxSize = createdWin.winSize;
            createdWin.SetWindowTitle(windowTitle);

            return createdWin;
        }

        public void ShowAt(Vector2 pos, bool centeredPosition)
        {
            if (centeredPosition)
            {
                pos.x -= winSize.x / 2;
                pos.y -= winSize.y / 2;
            }
            ShowUtility();
            this.SetSaveToLayout(false);
            position = new Rect(pos, winSize);
        }

        private void OnLostFocus()
        {
            Close();
        }

        private void OnGUI()
        {
            var e = Event.current;
            bool submit = e.type == EventType.KeyDown && (e.keyCode == KeyCode.KeypadEnter || e.keyCode == KeyCode.Return);

            var lineStyle = new GUIStyle();
            lineStyle.margin.top = 5;
            EditorGUILayout.BeginHorizontal(lineStyle);
            GUI.SetNextControlName("enteredText");
            enteredText = EditorGUILayout.TextField(enteredText);
            EditorGUILayout.EndHorizontal();

            if (!initialFocusWasSet)
            {
                EditorGUI.FocusTextInControl("enteredText");
                initialFocusWasSet = true;
            }

            var buttonStyle = new GUIStyle(GUI.skin.button);
            buttonStyle.fontSize = 12;
            buttonStyle.padding = new RectOffset(10, 10, 4, 4);
            EditorGUILayout.BeginHorizontal(lineStyle);
            submit = GUILayout.Button("Save", buttonStyle) || submit;
            if (submit)
            {
                if (saveFunction != null)
                    saveFunction.Invoke(enteredText);
                Close();
            }
            if (GUILayout.Button("Cancel", buttonStyle))
            {
                Close();
            }
            EditorGUILayout.EndHorizontal();
        }

        private void OnDestroy()
        {
            saveFunction = null;
        }
    }
}
