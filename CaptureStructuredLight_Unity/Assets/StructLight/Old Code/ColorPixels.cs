using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ColorPixels : MonoBehaviour
{
    public int screenwidth=1920;
    public int screenheight = 1080;
    public int startingMod = 20;
    Texture2D texture;
    Vector2Int pixelpos = new Vector2Int(0,0);
    // Start is called before the first frame update
    float prevtime;
    public double delay=0.01;
    RawImage theRawImg;
    void Start()
    {
   
        pixelpos.y = screenheight / 2;
      texture = new Texture2D(screenwidth, screenheight);
        GetComponent<RawImage>().material.mainTexture = texture;

        ColorThePixels(startingMod);
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            startingMod--;
            ColorThePixels(startingMod);
            print("mod = "+startingMod);
        }
        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            startingMod++;
            ColorThePixels(startingMod);
            print("mod = " + startingMod);
        }
        if(Time.unscaledTime-prevtime>delay) {
            pixelScanAll(startingMod);
            prevtime = Time.unscaledTime;
        }

    }
    void pixelScanAll(int mod)
    {
        //for (int y = 0; y < texture.height; y++)
        //{
        //    for (int x = 0; x < texture.width; x++)
        //    {
        //        Color color;
        //        //= ((x & y) != 0 ? Color.white : Color.gray);
        //        if (x % mod == 0)
        //        {
        //            color = Color.white;
        //            if (y % mod == 0)
        //            {
        //                color = Color.green;

        //            }
        //        }
        //        else { color = Color.black; }


        //        texture.SetPixel(x, y, color);
        //    }
        //}
        
        if (pixelpos.x > screenwidth)
        {
            pixelpos.x = 0;
            pixelpos.y++;
        }
        if (pixelpos.y > screenheight)
        {
            pixelpos.y = 0;
        }
        
        texture.SetPixel(pixelpos.x, pixelpos.y, Color.red);
        
        texture.SetPixel(pixelpos.x-1, pixelpos.y, Color.white);
        texture.SetPixel(pixelpos.x - 2, pixelpos.y, Color.white);
        texture.SetPixel(pixelpos.x + 1, pixelpos.y, Color.white);
        texture.SetPixel(pixelpos.x + 2, pixelpos.y, Color.white);

        //texture.SetPixel(pixelpos.x - 1, pixelpos.y-1, Color.white);
        //texture.SetPixel(pixelpos.x - 2, pixelpos.y-1, Color.white);
        //texture.SetPixel(pixelpos.x + 1, pixelpos.y+1, Color.white);
        //texture.SetPixel(pixelpos.x + 2, pixelpos.y+1, Color.white);
        texture.SetPixel(pixelpos.x , pixelpos.y - 1, Color.white);
        texture.SetPixel(pixelpos.x , pixelpos.y - 2, Color.white);
        texture.SetPixel(pixelpos.x , pixelpos.y + 1, Color.white);
        texture.SetPixel(pixelpos.x, pixelpos.y + 2, Color.white);

        pixelpos.x++;
        texture.Apply();


    }

    void ColorThePixels(int mod)
    {
        


        for (int y = 0; y < texture.height; y++)
        {
            for (int x = 0; x < texture.width; x++)
            {
                Color color;
                if (x % mod < mod / 2)
                {
                    color = Color.white;
                    if (y % mod < mod / 2)
                    {
                        color = Color.black;

                    }
                }
                else { color = Color.black; }

                texture.SetPixel(x, y, color);
            }
        }
        texture.Apply();
    }
}
