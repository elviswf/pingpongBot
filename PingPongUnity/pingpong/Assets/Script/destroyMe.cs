using UnityEngine;
using System.Collections;

public class destroyMe : MonoBehaviour {

	// Use this for initialization
	void Start () {
		StartCoroutine (DestroyMe()); 
	}
	
	// Update is called once per frame


	IEnumerator DestroyMe()
	{
		yield return new WaitForSeconds (0f);
		Destroy (gameObject); 
	}
}
