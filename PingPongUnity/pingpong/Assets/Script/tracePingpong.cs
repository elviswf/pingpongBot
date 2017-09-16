using UnityEngine;
using System.Collections;

public class tracePingpong : MonoBehaviour {

	// Use this for initialization
	private string [][]Array;  
	bool dataPrepared = false ;

	void Start ()  
	{  
		 	//读取csv二进制文件  
		    TextAsset binAsset = Resources.Load ("test", typeof(TextAsset)) as TextAsset;         
		          
		    //读取每一行的内容  
		    string [] lineArray = binAsset.text.Split ("\r"[0]);  
		          
		    //创建二维数组  
		    Array = new string [lineArray.Length][];  
			Debug.Log ("arrayLength: " + lineArray.Length);
		    //把csv中的数据储存在二位数组中  
		    for(int i =0; i < 47; i++)  
			{  
			    Array[i] = lineArray[i].Split (',');  
				Debug.Log (i + "_" + Array [i] [0] + " , " + Array [i] [1] + " , " + Array [i] [2] + " , " + Array [i] [3]);
			}  
			dataPrepared = true; 
	}  

	int dataIndex;
	float flyingTime = 0 ; 
	public GameObject pingpong; 
	// Update is called once per frame
	void Update () {
		if (dataPrepared) {
			flyingTime += Time.deltaTime; 
			if (dataIndex == 0) {
				pingpong.transform.position = new Vector3 (float.Parse(Array [dataIndex] [1]),float.Parse(Array [dataIndex] [3]),float.Parse(Array [dataIndex] [2])); 
				dataIndex++;
			}
			if ((dataIndex > 0) && (flyingTime >= 5 * float.Parse( Array [dataIndex - 1] [0] ) ) && (flyingTime <= 50 * float.Parse(Array [dataIndex + 1] [0]) ) && (dataIndex + 1 < 46)) {
				pingpong.transform.position = new Vector3 (float.Parse(Array [dataIndex] [1]),float.Parse(Array [dataIndex] [3])+100,(float)((float.Parse(Array [dataIndex] [2])-1370)*2.7)); 
				dataIndex++ ; 
				GameObject newObj =(GameObject) Instantiate (pingpong);
				newObj.AddComponent<destroyMe> (); 
				Debug.Log ("dataIndex : " + dataIndex);
			}

				
		}
	}
}
