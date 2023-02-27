/* 
 * Author:  Johanan Round
 */

using UnityEngine;
using System;
using System.Xml.Serialization;
using System.IO;

namespace EditorWindowFullscreen
{
    /// <summary>
    /// Serializes data to and from XML or JSON strings
    /// </summary>
    public static class SerializerUtility
    {
        public static T Deserialize<T>(string data)
        {
            object deserializedObj = default(T);
            try
            {
                if (data.Substring(0, 5) != "<?xml")
                {
                  //It's not an XML document, so attempt to parse json.
                  deserializedObj = JsonUtility.FromJson<T>(data);
                }
                else
                  deserializedObj = FromXml<T>(data);
            }
            catch (Exception e)
            {
                Debug.LogError(e);
            }
            return (T)deserializedObj;
        }

        public static string Serialize(object objectToSerialize)
        {
            return JsonUtility.ToJson(objectToSerialize);
        }

        public static T FromXml<T>(string xml)
        {
            XmlSerializer xmlSerializer = new XmlSerializer(typeof(T));
            object deserializedObj;
            using (var stringReader = new StringReader(xml))
            {
                deserializedObj = xmlSerializer.Deserialize(stringReader);
            }
            return (T)deserializedObj;
        }

        public static string ToXml(object objectToSerialize)
        {
            var xmlSerializer = new XmlSerializer(objectToSerialize.GetType());
            string serializedObj;
            using (var stringWriter = new StringWriter())
            {
                try
                {
                    xmlSerializer.Serialize(stringWriter, objectToSerialize);
                }
                catch (Exception e) {
                    Debug.LogError("Failed to serialize.\n" + e.Message + "\n\nPartial XML Serialization:\n" + stringWriter.ToString());
                }

                serializedObj = stringWriter.ToString();
            }
            return serializedObj;
        }
    }
}