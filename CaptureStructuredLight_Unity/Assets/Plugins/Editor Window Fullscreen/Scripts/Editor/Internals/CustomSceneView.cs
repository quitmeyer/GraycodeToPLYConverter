/* 
 * Author:  Johanan Round
 * Package: Editor Window Fullscreen
 * License: Unity Asset Store EULA (Editor extension asset. Requires 1 license per machine.)
 */

using UnityEngine;
using UnityEditor;
using System;
using System.Reflection;
using SceneViewState = UnityEditor.SceneView.SceneViewState;
using System.Linq;

namespace EditorWindowFullscreen
{
    class CustomSceneView : SceneView
    {
#pragma warning disable 0649
        public bool toolbarVisible = true;
#pragma warning restore 0649
        static MethodInfo baseOnGUI;
        static FieldInfo basePos;

        public static bool takeAudioState = true; //If true, takes the audio state of the last active scene view.
        private SceneView tookStateFromSceneView;
        private bool audioInitiallyEnabled;

        static CustomSceneView()
        {
            baseOnGUI = typeof(SceneView).GetMethod("OnGUI", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            basePos = typeof(EditorWindow).GetField("m_Pos", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        }

        public override void OnEnable()
        {
            var last = SceneView.lastActiveSceneView;

            var logEnabled = loggingEnabled;
            loggingEnabled = false;

            try
            {
                this.SetWindowTitle("Scene", true);
                base.OnEnable();

                var sceneIconContent = EditorGUIUtility.IconContent(typeof(SceneView).ToString());
                if (sceneIconContent != null)
                    this.titleContent.image = sceneIconContent.image;
            }
            catch (System.Exception e)
            {
                loggingEnabled = logEnabled;
                if (EWFDebugging.Enabled)
                {
                    Debug.LogException(e);
                    EWFDebugging.LogError(e.Message);
                }
            }
            finally
            {
                loggingEnabled = logEnabled;
            }

            if (last != null)
            {
                //Overwite the contents of the new scene view state with the last scene view state.
                tookStateFromSceneView = last;
#if UNITY_2019_1_OR_NEWER
                audioInitiallyEnabled = last.audioPlay;
#else
                audioInitiallyEnabled = last.m_AudioPlay;
#endif
                last.CopyStateTo(this);
            }
        }
#if UNITY_2018_1_OR_NEWER
#pragma warning disable 109
        protected new void OnGUI()
#pragma warning restore 109
#else
        void OnGUI()
#endif
        {
            var offsetToolbarHeight = EditorFullscreenState.sceneViewToolbarHeight + 1;
            var pos = this.basePosField;

            if (!toolbarVisible)
            {
                GUI.BeginGroup(new Rect(0, -offsetToolbarHeight, this.position.width, this.position.height + offsetToolbarHeight));

                //Trick the base OnGUI into drawing the Scene View at full size by temporarily increasing the window height.
                pos.height += offsetToolbarHeight;
                this.basePosField = pos;
            }

            try
            {
                baseOnGUI.Invoke(this, null);
            }
            catch (ExitGUIException e)
            {
                throw e;
            }
            catch (TargetInvocationException e)
            {
                if (e.InnerException is ExitGUIException)
                {
                    throw e.InnerException;
                }
                if (EWFDebugging.Enabled)
                {
                    Debug.LogException(e);
                    EWFDebugging.LogError(e.Message);
                }
            }

            if (!toolbarVisible)
            {
                //Reset the window height
                pos.height -= offsetToolbarHeight;
                this.basePosField = pos;

                GUI.EndGroup();
            }
        }

        new void OnDestroy()
        {
            //If the scene view which had its audio state taken still exists, re-enable the audio if it was originally enabled (Only one SceneView at a time can have audio enabled).
#if UNITY_2019_1_OR_NEWER
            if (takeAudioState && tookStateFromSceneView != null && audioInitiallyEnabled && this.audioPlay)
            {
                tookStateFromSceneView.audioPlay = audioInitiallyEnabled;
#else
            if (takeAudioState && tookStateFromSceneView != null && audioInitiallyEnabled && this.m_AudioPlay)
            {
                tookStateFromSceneView.m_AudioPlay = audioInitiallyEnabled;
#endif
                base.OnDestroy();
                try
                {
                    MethodInfo onFocusMethod = typeof(SceneView).GetMethod("OnFocus", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                    if (onFocusMethod != null) onFocusMethod.Invoke(tookStateFromSceneView, null);
                    else
                    {
                        tookStateFromSceneView.RefreshAudioPlay();
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
            }
            else
            {
                base.OnDestroy();
            }
        }

        private bool loggingEnabled
        {
            get
            {
#if UNITY_2017_2_OR_NEWER
                return Debug.unityLogger.logEnabled;
#elif UNITY_5_4_OR_NEWER || UNITY_5_3
                return Debug.logger.logEnabled;
#else
                return true;
#endif
            }
            set
            {
#if UNITY_2017_2_OR_NEWER
                Debug.unityLogger.logEnabled = value;
#elif UNITY_5_4_OR_NEWER || UNITY_5_3
                Debug.logger.logEnabled = value;
#else
                //Do nothing
#endif
            }
        }

        private Rect basePosField
        {
            get { return (Rect)basePos.GetValue(this); }
            set { basePos.SetValue(this, value); }
        }

        public static Type GetWindowType()
        {
            return typeof(SceneView);
        }


    }

    public static partial class SceneViewExtensions
    {
        private static SceneViewState GetSceneViewState(this SceneView sceneView)
        {
            SceneViewState state = null;
            try
            {
                FieldInfo sceneViewState = typeof(SceneView).GetField("m_SceneViewState", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                state = (SceneViewState)sceneViewState.GetValue(sceneView);
            }
            catch (System.Exception e)
            {
                if (EWFDebugging.Enabled)
                {
                    Debug.LogException(e);
                    EWFDebugging.LogError(e.Message);
                }
            }
            return state;
        }

        private static SceneViewState SetSceneViewState(this SceneView sceneView, SceneViewState newState)
        {
            SceneViewState state = null;
            try
            {
                FieldInfo sceneViewState = typeof(SceneView).GetField("m_SceneViewState", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                sceneViewState.SetValue(sceneView, newState);
            }
            catch (System.Exception e)
            {
                if (EWFDebugging.Enabled)
                {
                    Debug.LogException(e);
                    EWFDebugging.LogError(e.Message);
                }
            }
            return state;
        }

        internal static void RefreshAudioPlay(this SceneView sceneView)
        {
            MethodInfo refreshAudioPlay = typeof(SceneView).GetMethod("RefreshAudioPlay", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (refreshAudioPlay != null) refreshAudioPlay.Invoke(sceneView, null);
        }

        /// <summary>
        /// Copy the state of the SceneView to another SceneView.
        /// </summary>
        /// <param name="from">The SceneView to copy from.</param>
        /// <param name="copyToSceneView">The SceneView to copy to.</param>
        /// /// <returns>True if the copy was completely successful</returns>
        public static bool CopyStateTo(this SceneView fromSceneView, SceneView copyToSceneView)
        {
            bool completeCopy = true;
            SceneViewState fromState = GetSceneViewState(fromSceneView);
            SceneViewState fromStateCopy = new SceneViewState(fromState);
            copyToSceneView.SetSceneViewState(fromStateCopy);

            var publicFields = typeof(SceneView).GetFields(BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Public).Where(field => (field.FieldType.IsValueType || field.FieldType == typeof(string))).ToList();
            var publicProperties = typeof(SceneView).GetProperties(BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Public).Where(prop => prop.PropertyType.IsArray == false && (prop.PropertyType.IsValueType || prop.PropertyType == typeof(string)) && prop.GetGetMethod() != null && prop.GetSetMethod() != null).ToList();

            //Copy the Scene Visibility and Tools Popup status
            completeCopy = completeCopy && fromSceneView.CopyPrivateFieldTo("m_SceneVisActive", copyToSceneView);
            completeCopy = completeCopy && fromSceneView.CopyPrivateFieldTo("m_ShowContextualTools", copyToSceneView);

            //Copy the camera settings
#if UNITY_2018_1_OR_NEWER
            fromSceneView.CopyCameraSettingsTo(copyToSceneView);
#endif

            foreach (var field in publicFields)
            {
                if (CustomSceneView.takeAudioState == false && field.Name == "m_AudioPlay") continue;
                field.SetValue(copyToSceneView, field.GetValue(fromSceneView));
            }
            foreach (var prop in publicProperties)
            {
                prop.SetValue(copyToSceneView, prop.GetValue(fromSceneView, null), null);
            }
            


            if (CustomSceneView.takeAudioState)
            {
                copyToSceneView.RefreshAudioPlay();
            }
            return completeCopy;
        }

        public static void CopyCameraSettingsTo(this SceneView fromSceneView, SceneView copyToSceneView)
        {
#if UNITY_2019_1_OR_NEWER
            var publicProperties = typeof(SceneView.CameraSettings).GetProperties(BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Public).Where(prop => prop.PropertyType.IsArray == false && (prop.PropertyType.IsValueType || prop.PropertyType == typeof(string)) && prop.GetGetMethod() != null && prop.GetSetMethod() != null).ToList();
            foreach (var prop in publicProperties)
            {
                prop.SetValue(copyToSceneView.cameraSettings, prop.GetValue(fromSceneView.cameraSettings, null), null);
            }
#endif
        }

        private static bool CopyPrivateFieldTo(this SceneView fromSceneView, string fieldName, SceneView copyToSceneView)
        {
            var field = typeof(SceneView).GetField(fieldName, BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.NonPublic);
            if (field != null)
            {
                field.SetValue(copyToSceneView, field.GetValue(fromSceneView));
                return true; //Copy success
            }
            else return false;
        }
    }
}