/* 
 * Author:  Johanan Round
 * Package: Editor Window Fullscreen
 * License: Unity Asset Store EULA (Editor extension asset. Requires 1 license per machine.)
 */

using System;
using System.IO;
using System.Reflection;
using System.Collections.Generic;

namespace EditorWindowFullscreen
{
    static class WindowLayoutUtility
    {
        internal static System.Type windowLayoutType;
        internal static string projectLibraryPath;

        static WindowLayoutUtility()
        {
            windowLayoutType = System.Type.GetType("UnityEditor.WindowLayout,UnityEditor");
            projectLibraryPath = Directory.GetCurrentDirectory() + "/Library";
        }

        public static void SaveProjectLayout(string layoutFileName)
        {
            var filePath = Path.Combine(projectLibraryPath, layoutFileName);
            MethodInfo SaveWindowLayout = windowLayoutType.GetMethod("SaveWindowLayout", new[] { typeof(string) });
            SaveWindowLayout.Invoke(null, new[] { filePath });
        }

        public static bool LoadProjectLayout(string layoutFileName)
        {
            MethodInfo LoadWindowLayout;
            LoadWindowLayout = windowLayoutType.GetMethod("LoadWindowLayout", new[] { typeof(string), typeof(bool) });
            var filePath = Path.Combine(projectLibraryPath, layoutFileName);
            if (File.Exists(filePath))
            {
                LoadWindowLayout.Invoke(null, new object[] { filePath, false });
                return true;
            }
            else
            {
                if (EWFDebugging.Enabled) UnityEngine.Debug.LogError(filePath + " does not exist.");
                return false;
            }
        }

        public static void ReloadDefaultWindowPrefs()
        {
            var LoadDWP = windowLayoutType.GetMethod("LoadDefaultWindowPreferences", new Type[] {});
            if (LoadDWP != null) LoadDWP.Invoke(null, new object[] { });
            else UnityEditorInternal.InternalEditorUtility.LoadDefaultLayout();
        }
    }
}