import React, { useState } from 'react';
import axios from 'axios';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';

export default function Dashboard() {
  const [userId, setUserId] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);

  const getRecommendations = async () => {
    try {
      const res = await axios.get(`http://localhost:8000/recommend/${userId}`);
      setRecommendations(res.data.recommendations);
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to fetch recommendations.');
    }
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const uploadData = async () => {
    if (!file) return alert('Please select a CSV file.');

    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);
    setUploadProgress(0);

    try {
      await axios.post('http://localhost:8000/upload-data', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => {
          const percent = Math.round((e.loaded * 100) / e.total);
          setUploadProgress(percent);
        },
      });
      alert('Data uploaded and model updated successfully.');
    } catch (err) {
      alert(err.response?.data?.detail || 'Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-2xl font-bold">FlickWise Recommendation Dashboard</h1>

      <Card>
        <CardContent className="space-y-4 p-4">
          <h2 className="text-xl font-semibold">Get Recommendations</h2>
          <Input
            placeholder="Enter User ID"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
          />
          <Button onClick={getRecommendations}>Recommend</Button>

          {recommendations.length > 0 && (
            <div className="pt-4">
              <h3 className="font-semibold">Top Recommendations:</h3>
              <ul className="list-disc ml-5">
                {recommendations.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="space-y-4 p-4">
          <h2 className="text-xl font-semibold">Upload New Dataset</h2>
          <Input type="file" accept=".csv" onChange={handleFileChange} />
          <Button onClick={uploadData} disabled={uploading}>
            Upload
          </Button>
          {uploading && <Progress value={uploadProgress} className="mt-2" />}
        </CardContent>
      </Card>
    </div>
  );
}
