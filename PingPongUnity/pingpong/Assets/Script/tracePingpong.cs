using UnityEngine;
using System.Collections;

public class tracePingpong : MonoBehaviour {

	// Use this for initialization
	private string [][]Array;  
	bool dataPrepared = false ;
    int index = 0;

	void Start ()  
	{  
		 	//读取csv二进制文件  
		    TextAsset binAsset = Resources.Load ("trace0", typeof(TextAsset)) as TextAsset;         
		          
		    //读取每一行的内容  
		    string [] lineArray = binAsset.text.Split ("\r"[0]);  
		          
		    //创建二维数组  
		    Array = new string [lineArray.Length][];  
			Debug.Log ("arrayLength: " + lineArray.Length);
		    //把csv中的数据储存在二位数组中  
		    for(int i =0; i < lineArray.Length; i++)  
			{  
			    Array[i] = lineArray[i].Split (',');  
				Debug.Log (i + "_" + Array [i] [0] + " , " + Array [i] [1] + " , " + Array [i] [2] + " , " + Array [i] [3]);
			}  
			dataPrepared = true; 
	}  

	int dataIndex;
    int sign = 1;
	float flyingTime = 0 ;
    float x, y, z, preTime;
    double vx, vy, vz;
    int count = 0;
    int flag = 0;
    int cur = 0;
	public GameObject pingpong; 
	// Update is called once per frame
	void Update () {
		if (dataPrepared) {
			flyingTime += Time.deltaTime; 
			if (dataIndex == 0) {
				pingpong.transform.position = new Vector3 (float.Parse(Array [dataIndex] [1]),float.Parse(Array [dataIndex] [3]),float.Parse(Array [dataIndex] [2])); 
				dataIndex++;
			}
			if ((dataIndex > 0) && (dataIndex + 1 < Array.Length) &&(flyingTime >= 1 * float.Parse( Array [dataIndex - 1] [0] ) ) && (flyingTime <= 25 * float.Parse(Array [dataIndex + 1] [0]) )) {
                preTime = flyingTime;
                pingpong.transform.position = new Vector3 ((float.Parse(Array [dataIndex] [1])-762)*2,float.Parse(Array [dataIndex] [3])+100,(float)((float.Parse(Array [dataIndex] [2])-1370)*2.7)); 
				dataIndex++ ; 
				GameObject newObj =(GameObject) Instantiate (pingpong);
				newObj.AddComponent<destroyMe> (); 
				Debug.Log ("dataIndex : " + dataIndex);
			}
            if(dataIndex+1>=Array.Length)
            {
                if (sign>0)
                {
                    x = float.Parse(Array[dataIndex - 1][1]);
                    y = float.Parse(Array[dataIndex - 1][3]);
                    z = float.Parse(Array[dataIndex - 1][2]);
                    if (cur == 0)
                    {
                        vx = -315.3441729943841;
                        vy = -155.1063882469431;
                        vz = 2838.4144932320864;
                        cur = 1 - cur;
                    }
                    else
                    {
                        if (flag == 0)
                            vy = vy * 1.023;
                        else
                            vy = vy / 1.010;
                    }
                    x = (float)(x + vx * ((float)flyingTime - preTime)/2);
                    y = (float)(y + vy * (flyingTime - preTime)/2);
                    z = (float)(z + vz * (flyingTime - preTime)/2);
                    float temp = y + 150;
                    if (temp < 100)
                    {
                        flag = 1;
                        temp = -(temp - 100) + 150;
                    }
                    if (count % 3 == 0)
                    {
                        pingpong.transform.position = new Vector3((x - 762) * 2, temp, (float)((z - 1370) * 2.7));
                        GameObject newObj = (GameObject)Instantiate(pingpong);
                        newObj.AddComponent<destroyMe>();
                    }
                    count++;
                    if(count>100)
                    {
                        count = 0;
                        sign = 1 - sign;
                        cur = 0;
                    }
                }
                else
                {
                
                    sign = 1 - sign;
                    index += 1;
                    string fileName = "trace" + index.ToString();
                    //读取csv二进制文件  
                    TextAsset binAsset = Resources.Load(fileName, typeof(TextAsset)) as TextAsset;

                    //读取每一行的内容  
                    string[] lineArray = binAsset.text.Split("\r"[0]);

                    //创建二维数组  
                    Array = new string[lineArray.Length][];
                    Debug.Log("arrayLength: " + lineArray.Length);
                    //把csv中的数据储存在二位数组中  
                    for (int i = 0; i < lineArray.Length; i++)
                    {
                        Array[i] = lineArray[i].Split(',');
                        Debug.Log(i + "_" + Array[i][0] + " , " + Array[i][1] + " , " + Array[i][2] + " , " + Array[i][3]);
                    }
                    dataIndex = 0;
                    flyingTime = 0;
                    flag = 0;
                    dataPrepared = true;
                    if (index >= 6)
                        index = -1;
                }
            }

				
		}
	}
}
